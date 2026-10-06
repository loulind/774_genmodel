"""Rule-based downstream recommendations and misuse warnings.

Each recommendation is derived from a specific measured quantity, and that
quantity is quoted, so the advice is auditable. The LLM report writer may
elaborate on these but is instructed not to contradict them.
"""

from __future__ import annotations

from .critic import primary_metric_value
from .schema import Plan

GENERAL_WARNINGS = [
    "Never use synthetic samples as test or validation data: performance must be measured "
    "on held-out real data.",
    "Do not estimate population quantities (prevalence, class proportions, effect sizes, "
    "differential-expression statistics) from synthetic samples; they reflect the generator, "
    "not new evidence.",
    "Label synthetic records as synthetic wherever they are stored or shared; never pool them "
    "with real data in a way that hides their origin.",
]


def recommend(plan: Plan, results: dict, profile) -> dict:
    uses, warnings, verdict = [], list(GENERAL_WARNINGS), ""
    task = plan.task.task_type

    if not plan.use_generative_model:
        alt = results.get("alternative", {})
        verdict = "Generative modeling not recommended for this objective."
        uses.append(plan.alternative_if_not or "Use a non-generative analysis.")
        if alt:
            uses.append(f"Baseline evidence: {alt.get('summary', '')}")
        return {"verdict": verdict, "uses": uses, "warnings": GENERAL_WARNINGS[:1]}

    primary = next(m.name for m in plan.models if m.role == "primary")
    gen = results.get("generative", {})
    mem = gen.get("models", {}).get(primary, {})

    if task in ("generation", "augmentation", "class_balancing"):
        u = gen.get("utility", {}).get("models", {}).get(primary, {})
        key = "target_class_recall_gain" if task == "class_balancing" else "macro_f1_gain"
        g = u.get(key) or u.get("macro_f1_gain")
        if g:
            what = "target-class recall" if key == "target_class_recall_gain" else "macro-F1"
            span = f"{g['gain']:+.3f} (95% CI {g['ci_low']:+.3f} to {g['ci_high']:+.3f})"
            if g["ci_low"] > 0:
                verdict = f"Synthetic augmentation reliably improves {what}: {span}."
                uses.append(f"Augment the real training set with {primary} samples at the "
                            f"planned ratio when training the downstream classifier.")
            elif g["ci_high"] < 0:
                verdict = f"Synthetic augmentation harms {what}: {span}."
                uses.append("Do not use the synthetic samples for classifier training.")
            else:
                verdict = (f"No reliable effect of augmentation on {what}: {span}; the interval "
                           "includes zero.")
                uses.append("Treat augmentation as optional: the evidence neither supports nor "
                            "rules out a benefit. Re-test on a larger real test set before relying "
                            "on it.")
            cw = u.get("vs_class_weighted_target_recall") or u.get("vs_class_weighted_macro_f1")
            if cw:
                if cw["ci_high"] < 0 or cw["gain"] <= 0:
                    uses.append(f"Class weighting matches or beats synthetic oversampling "
                                f"(difference {cw['gain']:+.3f}, 95% CI {cw['ci_low']:+.3f} to "
                                f"{cw['ci_high']:+.3f}); prefer the simpler fix.")
                else:
                    uses.append(f"Synthetic oversampling outperforms class weighting by "
                                f"{cw['gain']:+.3f} (95% CI {cw['ci_low']:+.3f} to "
                                f"{cw['ci_high']:+.3f}).")
        t = u.get("tstr")
        real = gen.get("utility", {}).get("real_only")
        if t and real:
            ratio = t["macro_f1"] / max(real["macro_f1"], 1e-9)
            if ratio >= 0.9:
                uses.append(f"A classifier trained on synthetic data alone reaches {100 * ratio:.0f}% "
                            "of the real-data macro-F1: the samples carry most of the class signal "
                            "and are suitable for pipeline prototyping, teaching and method "
                            "development where real data cannot be shared.")
            else:
                warnings.append(f"Synthetic-only training reaches {100 * ratio:.0f}% of real-data "
                                "macro-F1: synthetic data is not a substitute for real data.")
        if mem.get("memorization_flag"):
            warnings.append(f"Memorization flagged: {100 * mem['too_close_fraction']:.1f}% of "
                            "synthetic samples are closer to a training record than 95% of real "
                            "holdout records are. Do not release or share these samples.")
        elif mem:
            uses.append(f"No memorization signal (DCR ratio {mem.get('dcr_ratio', float('nan')):.2f}, "
                        f"{100 * mem.get('too_close_fraction', 0):.1f}% too close vs ~5% expected); "
                        "privacy still requires a formal guarantee (e.g. differential privacy) "
                        "before external release.")
        for c, v in mem.get("target_classes", {}).items():
            if v["n_real_train"] < 30:
                warnings.append(f"'{c}' synthetic samples derive from only {v['n_real_train']} real "
                                f"training examples (and {v['n_real_holdout']} held-out): use them "
                                "only to stabilise classifier training, never to characterise the "
                                "class.")

    elif task == "imputation":
        imp = results.get("imputation", {})
        mine = imp.get("methods", {}).get(primary, {})
        best = imp.get("best_baseline")
        if imp.get("model_beats_best_baseline"):
            verdict = (f"{primary} imputes held-out entries more accurately than the best "
                       f"baseline ({best}): RMSE {mine.get('masked_rmse', float('nan')):.3f} vs "
                       f"{imp['methods'][best]['masked_rmse']:.3f}.")
            uses.append("Use multiple imputations (20 draws) and combine analyses with Rubin's "
                        "rules, so imputation uncertainty propagates to estimates.")
        else:
            verdict = (f"The generative imputer does not beat {best} "
                       f"(RMSE {mine.get('masked_rmse', float('nan')):.3f} vs "
                       f"{imp['methods'][best]['masked_rmse']:.3f}); use {best}.")
            uses.append(f"Impute with {best} (multiple imputation via chained equations), "
                        "combined with Rubin's rules.")
        cov = mine.get("multiple_imputation_coverage")
        if cov is not None:
            uses.append(f"Nominal 90% imputation intervals from {primary} covered "
                        f"{100 * cov:.0f}% of hidden values.")
        warnings = ["Evaluation masks entries completely at random; if values are missing not "
                    "at random (e.g. tests ordered only when abnormal), every imputer here is "
                    "biased and a sensitivity analysis is required.",
                    "Do not treat imputed values as observed measurements in reporting."]

    elif task == "anomaly_detection":
        an = results.get("anomaly", {})
        methods = an.get("methods", {})
        best = max(methods, key=lambda n: methods[n]["auprc"]) if methods else None
        verdict = (f"Best AUPRC: {best} ({methods[best]['auprc']:.3f}; prevalence "
                   f"{an.get('prevalence', 0):.3f}).") if best else ""
        uses.append(f"Use {best} scores to rank cases for review; choose the review threshold "
                    "from the precision-recall curve for the workload you can afford.")
        warnings = ["Scores are relative rankings, not calibrated probabilities of disease.",
                    "Anomalies of a kind absent from training may score differently; monitor "
                    "drift and re-validate on new data before deployment."]

    elif task == "representation_learning":
        rep = results.get("representation", {}).get(primary, {})
        verdict = (f"Latent linear-probe accuracy {rep.get('latent', {}).get('linear_probe')} vs "
                   f"PCA {rep.get('pca', {}).get('linear_probe')}.")
        uses.append("Use the latent means as features for clustering or visualisation.")

    value = primary_metric_value(results, plan, primary)
    return {"verdict": verdict, "uses": uses, "warnings": warnings,
            "primary_model": primary, "primary_metric": plan.evaluation.primary_metric
            if plan.evaluation else None, "primary_metric_value": value}
