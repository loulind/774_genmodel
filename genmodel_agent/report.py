"""Final analysis report: Markdown + standalone HTML, plus CSV evaluation tables.

Structure is fixed and data-driven (tables come straight from the run record);
the interpretive prose comes from the LLM writer when available, otherwise from
deterministic templates that quote the same numbers. Either way, every number
in the report is read from ``evaluation.json``, ``decision_log.json`` or
``model_summary.json`` -- nothing is typed in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .llm import LLMUnavailable, extract_json
from .planner.llm import load_prompt

FIDELITY_COLS = ["precision", "recall", "density", "coverage", "frechet_distance", "mmd",
                 "marginal_ks_mean", "numeric_ks_mean", "categorical_tv_mean",
                 "correlation_mean_abs_diff", "dcr_ratio", "too_close_fraction",
                 "mannwhitney_p_synth_closer", "memorization_flag", "n_synth_eval"]
WRITER_KEYS = ["executive_summary", "data_and_objective", "decisions_explained",
               "results_interpretation", "critic_and_revision", "downstream_recommendations",
               "misuse_warnings", "limitations"]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fidelity_table(gen: dict) -> pd.DataFrame:
    rows = {"real holdout, IID (reference)": gen.get("reference_real_holdout", {})}
    if gen.get("reference_real_test"):
        rows["real test split (shift diagnostic)"] = gen["reference_real_test"]
    rows.update(gen.get("models", {}))
    df = pd.DataFrame({k: {c: v.get(c) for c in FIDELITY_COLS} for k, v in rows.items()}).T
    return df.dropna(axis=1, how="all")


def utility_table(util: dict) -> pd.DataFrame:
    rows = {"real only": util["real_only"]}
    if util.get("class_weighted"):
        rows["real only, class-weighted"] = util["class_weighted"]
    for n, r in util["models"].items():
        if "augmented" in r:
            rows[f"real + {n}"] = r["augmented"]
        if "tstr" in r:
            rows[f"{n} only (TSTR)"] = r["tstr"]
    return pd.DataFrame({k: {c: v.get(c) for c in ("macro_f1", "macro_f1_sd", "balanced_accuracy",
                                                    "target_class_recall", "n_seeds")}
                         for k, v in rows.items()}).T


def gains_table(util: dict) -> pd.DataFrame:
    rows = {}
    for n, r in util["models"].items():
        for key, label in (("macro_f1_gain", "augmented - real: macro-F1"),
                           ("target_class_recall_gain", "augmented - real: target recall"),
                           ("vs_class_weighted_macro_f1", "augmented - class-weighted: macro-F1"),
                           ("vs_class_weighted_target_recall",
                            "augmented - class-weighted: target recall"),
                           ("tstr_vs_real_macro_f1", "synthetic-only - real: macro-F1")):
            if key in r:
                rows[f"{n}: {label}"] = r[key]
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def per_class_table(gen: dict) -> pd.DataFrame:
    rows = {}
    for n, r in gen.get("models", {}).items():
        for c, v in r.get("target_classes", {}).items():
            rows[f"{n}: {c}"] = v
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def write_tables(results: dict, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    gen = results.get("generative")
    if gen:
        fidelity_table(gen).to_csv(d / "fidelity.csv")
        pc = per_class_table(gen)
        if len(pc):
            pc.to_csv(d / "target_class_fidelity.csv")
        if gen.get("utility"):
            utility_table(gen["utility"]).to_csv(d / "utility.csv")
            gains_table(gen["utility"]).to_csv(d / "utility_gains.csv")
    for key in ("imputation", "anomaly"):
        if key in results:
            pd.DataFrame(results[key]["methods"]).T.to_csv(d / f"{key}.csv")
    if "representation" in results:
        pd.json_normalize(results["representation"]).T.to_csv(d / "representation.csv")
    if "alternative" in results:
        pd.Series(results["alternative"]).to_csv(d / "alternative_baseline.csv")


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or not len(df):
        return "_(none)_"
    def fmt(v):
        if isinstance(v, bool) or v is None:
            return "" if v is None else str(v)
        if isinstance(v, float):
            return f"{v:.{digits}g}" if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0) else f"{v:.{digits}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| | " + " | ".join(map(str, cols)) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------

def _record(agent) -> dict:
    """Everything the writer may cite (training curves dropped for size)."""
    return {
        "objective": agent.objective, "profile": agent.profile.model_dump(exclude={"source"}),
        "plan": agent.plan.model_dump(),
        "decision_log": [e.model_dump() for e in agent.log.entries],
        "training": {n: m.summary() for n, m in agent.models.items()},
        "evaluation": agent.results, "critique": getattr(agent, "critique", None),
        "revision": agent.revision, "rule_based_recommendations": agent.recs,
        "compute": {"device": agent.hw.device, "quick": agent.cfg.quick},
    }


def _template_sections(agent) -> dict:
    p, plan, recs = agent.profile, agent.plan, agent.recs
    primary = next((m for m in plan.models if m.role == "primary"), None)
    s = {}
    s["executive_summary"] = (
        f"Objective: *{agent.objective}* The agent profiled the data as {p.brief()} and "
        f"interpreted the task as **{plan.task.task_type}**. "
        + (f"It trained **{primary.name}** (primary) "
           f"{'and ' + ', '.join(m.name for m in plan.models if m.role != 'primary') + ' (comparison) ' if len(plan.models) > 1 else ''}"
           f"and evaluated them with primary metric `{plan.evaluation.primary_metric}`. "
           if primary else "It declined to use a generative model. ")
        + recs.get("verdict", ""))
    s["data_and_objective"] = "\n".join(
        f"- {e.decision}: **{e.chosen}** -- {e.rationale}" for e in agent.log.entries
        if e.step in ("profile", "objective", "preprocessing"))
    s["decisions_explained"] = "\n".join(
        f"- **{e.step} / {e.decision}: {e.chosen}.** {e.rationale}"
        + ("".join(f" Rejected *{a.option}*: {a.reason.rstrip('.')}." for a in e.alternatives_rejected))
        for e in agent.log.entries if e.step in ("warrant", "model", "evaluation", "sampling"))
    s["results_interpretation"] = (
        "Synthetic metrics are read against the real-holdout reference row: a synthetic set "
        "that scores like fresh real data is indistinguishable on that metric. See the tables "
        "below. " + recs.get("verdict", ""))
    rev = agent.revision
    s["critic_and_revision"] = (
        (getattr(agent, "critique", None).assessment if getattr(agent, "critique", None) else
         "No critic step (generation declined).")
        + (f" Revision `{rev['action']} {rev['target']}` changed the primary metric from "
           f"{rev['before']} to {rev['after']} and was **{'kept' if rev['kept'] else 'reverted'}**."
           if rev.get("ran") else " No revision was run."))
    s["downstream_recommendations"] = "\n".join(f"- {u}" for u in recs.get("uses", []))
    s["misuse_warnings"] = "\n".join(f"- {w}" for w in recs.get("warnings", []))
    lim = []
    if agent.cfg.quick:
        lim.append("Quick (smoke-test) mode: models were trained for seconds; results are not "
                   "meaningful estimates of model quality.")
    for n, m in agent.models.items():
        h = m.history
        if h.get("stopped_because") == "time":
            lim.append(f"{n} stopped on the time cap ({h['seconds']}s, {h['epochs_run']} epochs); "
                       "more training might improve it.")
    lim.append("Fidelity metrics depend on the evaluation feature space; the reference row "
               "calibrates them but does not make them absolute.")
    s["limitations"] = "\n".join(f"- {x}" for x in lim)
    return s


def narrative(agent) -> tuple[dict, str]:
    if agent.client.available:
        try:
            resp = agent.client.complete("writer", load_prompt("writer"),
                                         "Write the report sections for this run.\n\n"
                                         + json.dumps(_record(agent), indent=1, default=str))
            sections = extract_json(resp.text)
            if all(isinstance(sections.get(k), str) and sections[k].strip() for k in WRITER_KEYS):
                return sections, f"LLM ({resp.model_used})"
        except (LLMUnavailable, ValueError):
            pass
    return _template_sections(agent), "rule-based templates"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def write_report(agent, manifest: dict, narrative_result=None) -> None:
    sections, author = narrative_result or narrative(agent)
    p, plan, out = agent.profile, agent.plan, agent.out
    md = [f"# Generative modeling report: {p.name}", "",
          f"*Objective:* {agent.objective}  ",
          f"*Run:* {manifest['created']} on {agent.hw.device} ({agent.hw.device_name}); "
          f"{'quick' if agent.cfg.quick else 'full'} mode; seed {agent.cfg.seed}; plan source "
          f"**{plan.source}**; narrative by {author}; total time "
          f"{manifest['seconds_total'] / 60:.1f} min.", "",
          "## 1. Executive summary", sections["executive_summary"], "",
          "## 2. Data and objective", sections["data_and_objective"], ""]
    prof = {"modality": p.modality, "n samples": p.n_samples, "n features": p.n_features,
            "feature types (cont/disc/bin/cat)": f"{p.n_continuous}/{p.n_discrete}/{p.n_binary}/{p.n_categorical}",
            "classes": None if not p.labels else f"{p.labels.n_classes} (imbalance {p.labels.imbalance_ratio:.1f}, smallest '{p.labels.minority_class}' n={p.labels.minority_count})",
            "missing cells": f"{100 * p.missing.overall_fraction:.1f}%",
            "zero fraction": f"{100 * p.zero_fraction:.1f}%",
            "PCA components for 90% variance": p.pca_components_90,
            "near-duplicate fraction": p.near_duplicate_fraction,
            "predefined splits": p.predefined_splits}
    if p.counts:
        prof["NB vs ZINB (genes preferring ZINB; median pi)"] = \
            f"{p.counts.fraction_genes_zinb_preferred:.2f}; {p.counts.median_zero_inflation_pi:.3f}"
    md += ["| profile | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in prof.items()] + [""]

    md += ["## 3. How the agent decided", sections["decisions_explained"], "",
           "Full decision log: `decision_log.json`. Summary:", "",
           "| step | decision | chosen | source |", "|---|---|---|---|"]
    for e in agent.log.entries:
        md.append(f"| {e.step} | {e.decision[:60]} | {str(e.chosen)[:70].replace('|', '/')} | {e.source} |")
    md.append("")

    if plan.use_generative_model:
        md += ["## 4. Models and training", "",
               "| model | role | representation | conditional | parameters | epochs | time (s) | stopped | hyperparameters |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m in plan.models:
            if m.name not in agent.models:
                continue
            s = agent.models[m.name].summary()
            hp = ", ".join(f"{k}={v}" for k, v in m.hyperparameters.items())
            md.append(f"| {m.name} | {m.role} | {m.representation} | {m.conditional} | "
                      f"{s['n_parameters']:,} | {s.get('epochs_run')} | {s.get('seconds')} | "
                      f"{s.get('stopped_because')} | {hp} |")
        md += ["", f"![training curves](figures/{agent.figs['training']})" if "training" in agent.figs else "", ""]

        md += ["## 5. Evaluation", sections["results_interpretation"], ""]
        gen = agent.results.get("generative")
        if gen:
            md += [f"Feature space: {gen['feature_space']}. Real train n={gen['n_real_train_eval']}; "
                   f"reference holdout = {gen.get('holdout_definition', 'holdout')}, "
                   f"n={gen['n_real_holdout_eval']}; synthetic sets scored at the holdout's n.", ""]
            if gen.get("train_test_shift"):
                s = gen["train_test_shift"]
                md += [f"**Train/test shift:** Frechet distance of the test split is "
                       f"{s['frechet_ratio_test_vs_iid_holdout']:.2f}x that of IID holdout data "
                       f"-- {s['note']}.", ""]
            md += ["### Fidelity, diversity, memorization", "", _md(fidelity_table(gen)), ""]
            if gen.get("utility", {}).get("classifier"):
                md += [f"Downstream classifier: {gen['utility']['classifier']}.", ""]
            pc = per_class_table(gen)
            if len(pc):
                md += ["### Target classes", "", _md(pc), ""]
            if gen.get("utility"):
                md += ["### Downstream utility (real test set)", "", _md(utility_table(gen["utility"])),
                       "", "Gains with paired-bootstrap 95% intervals:", "",
                       _md(gains_table(gen["utility"])), ""]
        for key in ("imputation", "anomaly"):
            if key in agent.results:
                block = agent.results[key]
                md += [f"### {key.title()}", "", _md(pd.DataFrame(block["methods"]).T), "",
                       ", ".join(f"{k}: {v}" for k, v in block.items() if k != "methods"), ""]
        if "representation" in agent.results:
            md += ["### Representation", "", _md(pd.json_normalize(agent.results["representation"]).T), ""]
        for key, caption in (("samples", "Real (left block) vs synthetic samples per class"),
                             ("embedding", "Real vs synthetic in the evaluation feature space (PCA)"),
                             ("counts", "Per-gene statistics, synthetic vs real"),
                             ("marginals", "Marginal distributions, real vs synthetic"),
                             ("utility", "Downstream classifier by training condition"),
                             ("imputation", "Imputation error on held-out entries"),
                             ("anomaly", "Precision-recall on held-out data")):
            if key in agent.figs:
                md += [f"![{caption}](figures/{agent.figs[key]})", f"*{caption}.*", ""]
        md += ["## 6. Critic and revision", sections["critic_and_revision"], ""]
    else:
        alt = agent.results.get("alternative", {})
        md += ["## 4. Decision: no generative model", "",
               f"**Reason.** {plan.decision_rationale}", "",
               f"**Recommended alternative.** {plan.alternative_if_not}", "",
               f"**Baseline evidence.** {alt.get('summary', 'n/a')}", ""]

    md += ["## 7. Recommended downstream use", sections["downstream_recommendations"], "",
           "### Do not use the generated data for", sections["misuse_warnings"], "",
           "## 8. Limitations", sections["limitations"], "",
           "## 9. Reproducibility", "",
           f"- Command: `{manifest['command']}`",
           f"- Git commit: `{manifest['git_sha']}`; Python {manifest['python']}; "
           + ", ".join(f"{k} {v}" for k, v in manifest["packages"].items()),
           f"- LLM: {'model ' + manifest['llm']['model_requested'] + ', ' + str(len(manifest['llm']['calls'])) + ' calls, $' + str(manifest['llm']['total_cost_usd']) if manifest['llm']['enabled'] else 'not used (rule-based planner, critic and writer)'}; "
           "responses cached under the run directory (`--replay` reproduces the run without LLM access).",
           "- Files: `decision_log.json`, `plan.json`, `profile.json`, `evaluation.json`, "
           "`tables/*.csv`, `model_summary.json`, `recommendations.json`, `synthetic_sample/`, "
           "`run_manifest.json`.", ""]
    text = "\n".join(md)
    (out / "report.md").write_text(text)
    (out / "report.html").write_text(to_html(text, f"Report: {p.name}"))


def to_html(md_text: str, title: str) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
:root {{ --surface:#fcfcfb; --text:#0b0b0b; --text-2:#52514e; --rule:#e4e3df; --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --surface:#1a1a19; --text:#ffffff; --text-2:#c3c2b7; --rule:#3a3a37; --accent:#3987e5; }} }}
body {{ background:var(--surface); color:var(--text); font:15px/1.55 -apple-system, "Segoe UI", sans-serif;
       max-width:1100px; margin:0 auto; padding:24px 16px; }}
h1,h2,h3 {{ line-height:1.25; }} h2 {{ border-bottom:1px solid var(--rule); padding-bottom:4px; margin-top:2em; }}
table {{ border-collapse:collapse; font-size:12.5px; display:block; overflow-x:auto; margin:8px 0; }}
th,td {{ border-bottom:1px solid var(--rule); padding:4px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--text-2); font-weight:600; }}
img {{ max-width:100%; height:auto; background:#fcfcfb; border-radius:4px; }}
code {{ font-size:12.5px; }} em {{ color:var(--text-2); }} a {{ color:var(--accent); }}
</style></head><body>{body}</body></html>"""
