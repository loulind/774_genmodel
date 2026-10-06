"""LLM planner: profile + objective + rule-based proposal -> validated Plan.

Flow (every branch is logged):
  1. The heuristic planner writes a proposal; its decisions enter the log as the
     baseline reasoning.
  2. The LLM may adopt, modify or reject any element, recording a decision for
     each change.
  3. The result is validated against the same guardrails as any plan. Errors are
     sent back once for repair; if the repaired plan still fails, the heuristic
     proposal is used and the log says why.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Budget
from ..decision_log import DecisionLog
from ..llm import LLMClient, LLMUnavailable, extract_json
from ..models.registry import MAX_MODELS_PER_PLAN, REGISTRY
from ..schema import DataProfile, Plan, TaskSpec
from . import heuristic
from .validator import (KNOWN_BASELINES, KNOWN_METRICS, KNOWN_PRIMARY, TOTAL_BUDGET_MULTIPLE,
                        validate)

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def library_summary() -> list[dict]:
    return [{"name": m.name, "family": m.family, "modalities": list(m.modalities),
             "tasks": list(m.tasks), "conditional_capable": m.conditional_capable,
             "representations": list(m.representations), "has_likelihood": m.has_likelihood,
             "hyperparameter_ranges": {k: list(v) for k, v in m.hp_ranges.items()},
             "summary": m.summary} for m in REGISTRY.values()]


def build_user_message(profile: DataProfile, task: TaskSpec, budget: Budget,
                       proposal: Plan, proposal_log: DecisionLog) -> str:
    cap = budget.train_seconds_image if profile.modality == "image" else budget.train_seconds_other
    payload = {
        "objective": task.raw_objective,
        "parsed_task": task.model_dump(),
        "data_profile": profile.model_dump(exclude={"source"}),
        "compute_budget": {"device": budget.device, "per_model_train_seconds": cap,
                           "total_train_seconds": TOTAL_BUDGET_MULTIPLE * cap,
                           "max_models": MAX_MODELS_PER_PLAN,
                           "max_synthetic_samples": budget.n_synthetic_max,
                           "max_train_samples": budget.max_train_samples},
        "model_library": library_summary(),
        "allowed_metrics": sorted(KNOWN_METRICS),
        "allowed_primary_metrics": sorted(KNOWN_PRIMARY),
        "allowed_baselines": sorted(KNOWN_BASELINES),
        "proposal": {"plan": proposal.model_dump(),
                     "reasoning": [{"step": d.step, "chosen": d.chosen, "rationale": d.rationale}
                                   for d in proposal_log.entries]},
    }
    return ("Plan the generative-modeling workflow for this dataset and objective.\n\n"
            + json.dumps(payload, indent=1, default=str))


def _log_llm_decisions(log: DecisionLog, decisions: list) -> None:
    for d in decisions or []:
        try:
            log.add(step=str(d.get("step", "llm")), decision=str(d.get("decision", "")),
                    chosen=str(d.get("chosen", "")), rationale=str(d.get("rationale", "")),
                    alternatives=[(str(a.get("option", "")), str(a.get("reason", "")))
                                  for a in d.get("alternatives_rejected", []) or []],
                    source="llm")
        except Exception:
            continue


def _diff(proposal: Plan, final: Plan) -> list[str]:
    out = []
    if proposal.use_generative_model != final.use_generative_model:
        out.append(f"use_generative_model {proposal.use_generative_model} -> "
                   f"{final.use_generative_model}")
    a = {(m.name, m.role) for m in proposal.models}
    b = {(m.name, m.role) for m in final.models}
    if a != b:
        out.append(f"models {sorted(a)} -> {sorted(b)}")
    for m in final.models:
        old = next((p for p in proposal.models if p.name == m.name), None)
        if old:
            changed = {k: (old.hyperparameters.get(k), v) for k, v in m.hyperparameters.items()
                       if old.hyperparameters.get(k) != v}
            if changed:
                out.append(f"{m.name} hyperparameters " + ", ".join(
                    f"{k}: {o} -> {n}" for k, (o, n) in changed.items()))
    if proposal.evaluation and final.evaluation and \
            proposal.evaluation.primary_metric != final.evaluation.primary_metric:
        out.append(f"primary metric {proposal.evaluation.primary_metric} -> "
                   f"{final.evaluation.primary_metric}")
    if proposal.sampling.n_per_class != final.sampling.n_per_class:
        out.append("sampling counts changed")
    return out


def plan(profile: DataProfile, task: TaskSpec, budget: Budget, log: DecisionLog,
         client: LLMClient) -> Plan:
    proposal_log = DecisionLog()
    proposal = heuristic.plan(profile, task, budget, proposal_log)
    log.extend(proposal_log.entries)

    def fallback(reason: str) -> Plan:
        log.add("planner", "planner used", "heuristic (fallback)", reason, [], {},
                source="system")
        checked = validate(proposal, profile, budget)
        for r in checked.repairs:
            log.add("validator", "plan repair", r, "guardrail", source="validator")
        out = checked.plan or proposal
        out.source = "heuristic_fallback" if client.available else "heuristic"
        return out

    if not client.available:
        return fallback("No LLM available (--no-llm or no backend); rule-based plan used.")

    system = load_prompt("planner")
    user = build_user_message(profile, task, budget, proposal, proposal_log)
    try:
        resp = client.complete("planner", system, user)
        raw = extract_json(resp.text)
    except (LLMUnavailable, ValueError) as err:
        return fallback(f"LLM planning failed: {err}")

    source = "llm"
    plan_dict = raw.get("plan", raw)
    plan_dict.setdefault("task", task.model_dump())
    plan_dict.pop("source", None)            # bookkeeping, set by us, not the model
    result = validate(plan_dict, profile, budget)
    if not result.ok:
        log.add("validator", "LLM plan validation", "failed; one repair attempt",
                "; ".join(result.errors), source="validator")
        repair_user = (user + "\n\nYour previous answer:\n" + resp.text[:20000] +
                       "\n\nIt failed validation with these errors:\n- " +
                       "\n- ".join(result.errors) +
                       "\n\nReturn the corrected JSON object only.")
        try:
            resp2 = client.complete("planner_repair", system, repair_user)
            raw = extract_json(resp2.text)
            plan_dict = raw.get("plan", raw)
            plan_dict.setdefault("task", task.model_dump())
            plan_dict.pop("source", None)
            result = validate(plan_dict, profile, budget)
            source = "llm_repaired"
        except (LLMUnavailable, ValueError) as err:
            return fallback(f"LLM plan invalid ({'; '.join(result.errors)}) and repair "
                            f"failed: {err}")
        if not result.ok:
            return fallback(f"LLM plan still invalid after one repair: "
                            f"{'; '.join(result.errors)}")

    final = result.plan
    final.source = source
    _log_llm_decisions(log, raw.get("decisions"))
    for r in result.repairs:
        log.add("validator", "plan repair", r, "LLM plan adjusted to satisfy a guardrail.",
                source="validator")
    changes = _diff(proposal, final)
    log.add("planner", "planner used", source,
            ("LLM revised the rule-based proposal: " + "; ".join(changes)) if changes
            else "LLM adopted the rule-based proposal unchanged.",
            [], {"llm_model": resp.model_used, "n_changes": len(changes)}, source="llm")
    return final
