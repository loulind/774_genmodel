"""Critic: read the results, decide whether one revision is worth running.

This is what distinguishes the agent from a fixed pipeline: the metrics can
implicate a choice the planner could not have known was wrong until a model ran.

Two layers:
  1. Deterministic triggers, each with a suggested remedy:
       memorization        synthetic samples closer to training data than real
                           holdout points are (overall, or within a target class)
       comparison_better   the comparison model beat the primary on the primary metric
       baseline_better     a classical baseline beat the generative model on the task
       low_diversity       recall/coverage far below the real-holdout reference
       utility_harm        augmentation made the classifier worse (interval below 0)
  2. A judgement -- the LLM's when available, otherwise the first actionable
     trigger by priority. The LLM's proposal is validated (model names and
     hyperparameter ranges) before anything runs.

Limits (as in Project 1): at most one revision per run, and a revision is kept
only if the primary metric improves -- otherwise the original result stands and
the log records both.
"""

from __future__ import annotations

import json

from .decision_log import DecisionLog
from .llm import LLMClient, LLMUnavailable, extract_json
from .models.registry import REGISTRY, clip_hyperparameters
from .planner.llm import library_summary, load_prompt
from .schema import Alternative, Critique, DataProfile, Plan, Trigger

PRIORITY = ["memorization", "comparison_better", "baseline_better", "low_diversity"]
LOW_DIVERSITY_RATIO = 0.5
TARGET_TOO_CLOSE = 0.2


# ---------------------------------------------------------------------------
# The primary metric, as "higher is better"
# ---------------------------------------------------------------------------

def primary_metric_value(results: dict, plan: Plan, model: str) -> float | None:
    """Return the plan's primary metric for ``model`` with lower-is-better metrics negated."""
    name = plan.evaluation.primary_metric if plan.evaluation else None
    try:
        if name in ("macro_f1_gain", "target_class_recall_gain"):
            return float(results["generative"]["utility"]["models"][model][name]["gain"])
        if name in ("frechet_distance", "mmd"):
            return -float(results["generative"]["models"][model][name])
        if name in ("precision_recall",):
            r = results["generative"]["models"][model]
            return 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9)
        if name == "masked_rmse":
            return -float(results["imputation"]["methods"][model]["masked_rmse"])
        if name in ("auprc", "auroc"):
            return float(results["anomaly"]["methods"][model][name])
        if name == "linear_probe":
            return float(results["representation"][model]["latent"]["linear_probe"])
    except (KeyError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Deterministic triggers
# ---------------------------------------------------------------------------

def _retune_for_memorization(spec) -> dict:
    hp = spec.hyperparameters
    if spec.name in ("vae", "nb_vae", "zinb_vae", "masked_vae"):
        return {"weight_decay": max(1e-4, 10 * float(hp.get("weight_decay", 0) or 1e-5)),
                "latent_dim": max(2, int(hp.get("latent_dim", 8)) // 2)}
    if spec.name in ("ddpm_unet", "ddpm_mlp"):
        return {"weight_decay": max(1e-4, 10 * float(hp.get("weight_decay", 0) or 1e-5)),
                "max_epochs": max(10, int(hp.get("max_epochs", 200)) // 2)}
    return {"weight_decay": 1e-4}


def find_triggers(plan: Plan, results: dict) -> list[Trigger]:
    triggers: list[Trigger] = []
    primary = next((m for m in plan.models if m.role == "primary"), None)
    if primary is None:
        return triggers
    others = [m for m in plan.models if m.role != "primary"]
    gen = results.get("generative", {})
    ref = gen.get("reference_real_holdout", {})

    for spec in plan.models:
        r = gen.get("models", {}).get(spec.name, {})
        flagged = r.get("memorization_flag")
        close_classes = [c for c, v in r.get("target_classes", {}).items()
                         if v.get("too_close_fraction", 0) > TARGET_TOO_CLOSE and v["n_synth"] >= 10]
        if flagged or close_classes:
            clean = [o for o in plan.models if o.name != spec.name and not
                     gen.get("models", {}).get(o.name, {}).get("memorization_flag")]
            what = (f"too_close_fraction {r.get('too_close_fraction', 0):.2f} (expected ~0.05)"
                    if flagged else f"target classes {close_classes} reproduce training cells")
            if spec.role == "primary" and clean:
                triggers.append(Trigger(code="memorization", model=spec.name,
                                        message=f"{spec.name}: {what}",
                                        suggested_action="switch", suggested_target=clean[0].name))
            else:
                triggers.append(Trigger(code="memorization", model=spec.name,
                                        message=f"{spec.name}: {what}",
                                        suggested_action="retune", suggested_target=spec.name,
                                        suggested_changes=_retune_for_memorization(spec)))

        if ref.get("recall") and r.get("recall") is not None and \
                r["recall"] < LOW_DIVERSITY_RATIO * ref["recall"]:
            changes = {}
            if spec.name.startswith("ddpm") and float(spec.hyperparameters.get("guidance_scale", 0)) > 0:
                changes = {"guidance_scale": round(float(spec.hyperparameters["guidance_scale"]) / 2, 2)}
            elif spec.name == "vae":
                changes = {"beta": round(float(spec.hyperparameters.get("beta", 1.0)) / 2, 3)}
            triggers.append(Trigger(
                code="low_diversity", model=spec.name,
                message=f"{spec.name}: recall {r['recall']:.3f} < {LOW_DIVERSITY_RATIO} x "
                        f"real-holdout recall {ref['recall']:.3f}",
                suggested_action="retune" if changes else "none",
                suggested_target=spec.name, suggested_changes=changes))

    p_val = primary_metric_value(results, plan, primary.name)
    for o in others:
        o_val = primary_metric_value(results, plan, o.name)
        if p_val is not None and o_val is not None and o_val > p_val + 0.01 * max(1, abs(p_val)):
            triggers.append(Trigger(
                code="comparison_better", model=o.name,
                message=f"{o.name} beats primary {primary.name} on "
                        f"{plan.evaluation.primary_metric} ({o_val:.4g} vs {p_val:.4g}, "
                        f"higher is better)",
                suggested_action="switch", suggested_target=o.name))

    if gen.get("utility", {}).get("baseline_degenerate"):
        triggers.append(Trigger(code="degenerate_baseline", model=primary.name,
                                message="real-only baseline classifier is at chance level; "
                                        "utility gains are not interpretable"))
    util = gen.get("utility", {}).get("models", {}).get(primary.name, {})
    g = util.get("macro_f1_gain")
    if g and g["ci_high"] < 0:
        triggers.append(Trigger(code="utility_harm", model=primary.name,
                                message=f"augmentation lowers macro-F1 by {-g['gain']:.3f} "
                                        f"(95% CI {g['ci_low']:.3f} to {g['ci_high']:.3f})"))

    for key, metric, lower_better in (("imputation", "masked_rmse", True),
                                      ("anomaly", "auprc", False)):
        block = results.get(key)
        if not block:
            continue
        methods = block["methods"]
        mine = methods.get(primary.name, {}).get(metric)
        base = {n: v.get(metric) for n, v in methods.items()
                if n not in {m.name for m in plan.models} and v.get(metric) is not None}
        if mine is None or not base:
            continue
        best_name = (min if lower_better else max)(base, key=base.get)
        better = base[best_name] < mine if lower_better else base[best_name] > mine
        if better:
            hp = primary.hyperparameters
            if primary.name == "masked_vae":
                changes = {"latent_dim": max(2, int(hp.get("latent_dim", 8)) // 2),
                           "weight_decay": min(1e-2, 10 * float(hp.get("weight_decay", 1e-5)))}
            elif primary.name in ("maf", "realnvp"):
                changes = {"noise_std": min(0.5, max(0.05, 2 * float(hp.get("noise_std", 0)))),
                           "n_blocks": max(2, int(hp.get("n_blocks", 5)) - 2)}
            else:
                changes = {}
            triggers.append(Trigger(
                code="baseline_better", model=primary.name,
                message=f"baseline {best_name} beats {primary.name} on {metric} "
                        f"({base[best_name]:.4g} vs {mine:.4g})",
                suggested_action="retune" if changes else "none",
                suggested_target=primary.name, suggested_changes=changes))
    return triggers


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------

def _rule_critique(triggers: list[Trigger], primary: str | None = None) -> Critique:
    """One revision is available, so spend it on the model whose results the
    recommendations rest on: retunes of a comparison model are logged, not run."""
    for code in PRIORITY:
        for t in triggers:
            if t.code == code and t.suggested_action != "none" and \
                    (primary is None or t.suggested_action == "switch" or
                     t.suggested_target == primary):
                return Critique(
                    assessment=f"Rule-based critic: trigger '{t.code}' -- {t.message}.",
                    action=t.suggested_action, target_model=t.suggested_target,
                    hyperparameter_changes=t.suggested_changes,
                    rationale=f"Highest-priority actionable trigger ({' > '.join(PRIORITY)}).",
                    alternatives_rejected=[Alternative(option="no revision",
                                                       reason=t.message)],
                    expected_effect="primary metric improves", triggers=triggers)
    msg = "; ".join(t.message for t in triggers) or "no trigger fired"
    return Critique(assessment=f"Rule-based critic: no actionable problem ({msg}).",
                    action="none", rationale="No trigger with a concrete remedy.",
                    triggers=triggers)


def _validated(c: dict, plan: Plan, profile: DataProfile, triggers) -> Critique | None:
    action = c.get("action", "none")
    if action not in ("none", "retune", "switch"):
        return None
    target = c.get("target_model")
    changes = c.get("hyperparameter_changes") or {}
    if action != "none":
        info = REGISTRY.get(target or "")
        if info is None or profile.modality not in info.modalities or \
                plan.task.task_type not in info.tasks:
            return None
        changes, _ = clip_hyperparameters(target, changes)
        changes = {k: v for k, v in changes.items() if k in (c.get("hyperparameter_changes") or {})}
    return Critique(
        assessment=str(c.get("assessment", "")), action=action, target_model=target,
        hyperparameter_changes=changes, rationale=str(c.get("rationale", "")),
        alternatives_rejected=[Alternative(option=str(a.get("option", "")),
                                           reason=str(a.get("reason", "")))
                               for a in c.get("alternatives_rejected", []) or []],
        expected_effect=str(c.get("expected_effect", "")), triggers=triggers, source="llm")


VOLATILE = {"seconds"}


def critique(plan: Plan, profile: DataProfile, results: dict, training: dict,
             client: LLMClient, log: DecisionLog, budget: dict | None = None) -> Critique:
    triggers = find_triggers(plan, results)
    for t in triggers:
        log.add("critic", f"trigger: {t.code}", t.model, t.message, [],
                {"suggested_action": t.suggested_action, "suggested_changes": t.suggested_changes},
                source="critic")
    out = None
    if client.available:
        training = {n: {k: v for k, v in t.items() if k not in VOLATILE}
                    for n, t in training.items()}
        payload = {"objective": plan.task.raw_objective, "plan": plan.model_dump(),
                   "profile_summary": profile.brief(), "training": training,
                   "compute_budget": budget or {},
                   "results": results, "triggers": [t.model_dump() for t in triggers],
                   "model_library": library_summary()}
        try:
            resp = client.complete("critic", load_prompt("critic"),
                                   "Review this run.\n\n" + json.dumps(payload, indent=1,
                                                                     default=str))
            out = _validated(extract_json(resp.text), plan, profile, triggers)
            if out is None:
                log.add("critic", "LLM critique", "rejected",
                        "LLM proposed an invalid action/model; rule-based critic used.",
                        source="validator")
        except (LLMUnavailable, ValueError) as err:
            log.add("critic", "LLM critique", "unavailable", str(err), source="system")
    primary = next((m.name for m in plan.models if m.role == "primary"), None)
    out = out or _rule_critique(triggers, primary)
    log.add("critic", "revise the plan?",
            out.action if out.action == "none" else f"{out.action} {out.target_model} "
            f"{out.hyperparameter_changes or ''}".strip(),
            f"{out.assessment} {out.rationale}".strip(),
            [(a.option, a.reason) for a in out.alternatives_rejected],
            {"expected_effect": out.expected_effect}, source="critic")
    return out
