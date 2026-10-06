"""Build outputs/robustness/summary.{csv,md}: one row per dataset x objective.

Every value is read from a run's own output files (profile.json, plan.json,
evaluation.json, model_summary.json, recommendations.json, run_manifest.json),
so the table can be regenerated from the runs and nothing is transcribed by hand.

    uv run python scripts/summarize_robustness.py label=path[:mode] ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else {}


def highlights(pr: dict) -> str:
    out = [f"{pr['modality']}, n={pr['n_samples']:,}, p={pr['n_features']:,}"]
    lab = pr.get("labels")
    if lab:
        out.append(f"{lab['n_classes']} classes, imbalance {lab['imbalance_ratio']:.1f} "
                   f"(min n={lab['minority_count']})")
    if pr["missing"]["overall_fraction"] > 0:
        out.append(f"{100 * pr['missing']['overall_fraction']:.1f}% missing")
    if pr["zero_fraction"] > 0.5:
        out.append(f"{100 * pr['zero_fraction']:.0f}% zeros")
    if pr.get("counts") and pr["counts"].get("median_zero_inflation_pi") is not None:
        out.append(f"ZINB pi {pr['counts']['median_zero_inflation_pi']:.3f}")
    nd = max(pr.get("duplicate_fraction") or 0, pr.get("near_duplicate_fraction") or 0)
    if nd > 0.05:
        out.append(f"{100 * nd:.0f}% (near-)duplicates")
    if pr.get("n_categorical"):
        out.append(f"{pr['n_categorical']} categorical cols")
    return "; ".join(out)


def main_metrics(ev: dict, plan: dict) -> str:
    if "alternative" in ev:
        return ev["alternative"].get("summary", "")
    primary = next((m["name"] for m in plan["models"] if m["role"] == "primary"), None)
    if "generative" in ev:
        g = ev["generative"]
        ref, m = g.get("reference_real_holdout", {}), g["models"].get(primary, {})
        parts = [f"FD {m.get('frechet_distance', float('nan')):.2f} (real ref "
                 f"{ref.get('frechet_distance', float('nan')):.2f})",
                 f"recall {m.get('recall', float('nan')):.2f} (ref {ref.get('recall', float('nan')):.2f})",
                 f"too-close {100 * m.get('too_close_fraction', 0):.1f}%"]
        u = g.get("utility", {}).get("models", {}).get(primary, {})
        key = plan["evaluation"]["primary_metric"]
        if key in u:
            x = u[key]
            parts.append(f"{key} {x['gain']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]")
        if "tstr" in u:
            parts.append(f"TSTR macro-F1 {u['tstr']['macro_f1']:.3f} vs real "
                         f"{g['utility']['real_only']['macro_f1']:.3f}")
        return "; ".join(parts)
    if "imputation" in ev:
        meth = ev["imputation"]["methods"]
        return "masked RMSE " + ", ".join(f"{n} {v['masked_rmse']:.3f}" for n, v in meth.items()
                                          if v.get("masked_rmse") is not None)
    if "anomaly" in ev:
        meth = ev["anomaly"]["methods"]
        return (f"AUPRC (prevalence {ev['anomaly']['prevalence']:.3f}): "
                + ", ".join(f"{n} {v['auprc']:.3f}" for n, v in meth.items())
                + "; AUROC " + ", ".join(f"{n} {v['auroc']:.3f}" for n, v in meth.items()))
    return ""


def row(label: str, d: Path, mode: str) -> dict:
    pr, plan, ev = load(d, "profile.json"), load(d, "plan.json"), load(d, "evaluation.json")
    ms, rec, man = load(d, "model_summary.json"), load(d, "recommendations.json"), load(d, "run_manifest.json")
    models = plan.get("models", [])
    primary = next((m for m in models if m["role"] == "primary"), None)
    keys = ("latent_dim", "base_channels", "guidance_scale", "n_blocks", "hidden_dim",
            "noise_std", "beta", "batch_size", "n_importance")
    rev = ms.get("revision", {})
    crit = ms.get("critique") or {}
    return {
        "dataset": label, "run": mode, "objective": plan["task"]["raw_objective"],
        "profile highlights": highlights(pr),
        "task (parsed)": plan["task"]["task_type"],
        "generative model used?": "yes" if plan["use_generative_model"] else "no (declined)",
        "models (primary; comparison)": "; ".join(f"{m['name']}{' (cond.)' if m['conditional'] else ''}"
                                                   for m in models) or "-",
        "key hyperparameters (primary)": ", ".join(f"{k}={primary['hyperparameters'][k]}"
                                                   for k in keys if primary and k in primary["hyperparameters"]) or "-",
        "primary metric": (plan.get("evaluation") or {}).get("primary_metric", "-"),
        "main results": main_metrics(ev, plan),
        "recommendation": rec.get("verdict", ""),
        "critic": crit.get("action", "-") if crit else "-",
        "revision": ("kept" if rev.get("kept") else "reverted") + f" ({rev['action']} {rev['target']})"
        if rev.get("ran") else "none",
        "plan source": plan.get("source", ""),
        "LLM cost (USD)": man.get("llm", {}).get("total_cost_usd"),
        "runtime (s)": man.get("seconds_total"),
    }


def main(argv: list[str]) -> None:
    rows = []
    for arg in argv:
        label, spec = arg.split("=", 1)
        path, _, mode = spec.partition(":")
        rows.append(row(label, ROOT / path, mode or "quick"))
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    md = ["# Robustness summary: agent decisions across datasets", "",
          "One row per dataset x objective. Robustness rows ran in `--quick` mode (seconds of "
          "training per model): their metrics check that the pipeline and the decisions work, "
          "not model quality. The two main datasets are included from their full runs "
          "(`outputs/generated_report_1`, `outputs/generated_report_2`) for comparison. All "
          "values are read from each run's output files by `scripts/summarize_robustness.py`.", "",
          "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v).replace("|", "/") for v in r.values) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(df[["dataset", "task (parsed)", "generative model used?", "models (primary; comparison)",
              "critic", "revision"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
