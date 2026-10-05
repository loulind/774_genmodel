"""Deterministic planner: profile + objective + budget -> Plan.

This is both the ``--no-llm`` planner and the fallback when an LLM plan cannot
be repaired. Every rule reads only the profile, the TaskSpec and the compute
budget -- never a dataset name -- and every decision is written to the decision
log with the numbers that triggered it and the options it rejected.

Thresholds are named constants so the report can quote them, and each carries
the reasoning for its value.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Budget
from ..decision_log import DecisionLog
from ..models.registry import MAX_FLOW_DIM
from ..schema import (
    DataPlan, DataProfile, EvalPlan, ModelSpec, Plan, SamplingPlan, TaskSpec,
)

# --- warrant thresholds ------------------------------------------------------
# A class with fewer training examples than this cannot be modelled without the
# generator effectively copying its members; we refuse to synthesise it.
MIN_CLASS_TRAIN = 10
# Below this largest/smallest ratio, class weights or threshold tuning already
# fix any imbalance; synthetic oversampling adds variance and no information.
MILD_IMBALANCE = 3.0
# "Low-dimensional tabular": a regularised classifier is already well-specified
# here, and a generator fitted to the same rows cannot add information about
# P(y | x) that the classifier lacks.
LOW_DIM_TABULAR = 100
# Image sets smaller than this favour a VAE over diffusion: diffusion models are
# documented to memorise small training sets (Somepalli et al. 2023).
SMALL_IMAGE_TRAIN = 2_000
# Mixed-type tables smaller than this favour a likelihood-based VAE (TVAE) over
# CTGAN; Xu et al. (2019) report TVAE competitive or better on small tables.
SMALL_TABLE_TRAIN = 1_000
# Synthetic-to-real caps. Beyond ~10x, extra synthetic minority samples are
# draws from a smoothed copy of a handful of real points, not new evidence.
MAX_SYNTH_PER_REAL = 10
AUGMENT_RATIO = 1.0
# Flows on data with repeated rows: above this duplicate share, train on
# x + N(0, DEQUANTISE_STD^2) in standardised units.
DUPLICATE_TOLERANCE = 0.01
DEQUANTISE_STD = 0.05


def _pow2(x: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, 2 ** round(math.log2(max(x, 1))))))


class Context:
    """Derived quantities the rules share (training sizes after splits/budget)."""

    def __init__(self, profile: DataProfile, task: TaskSpec, budget: Budget):
        self.profile, self.task, self.budget = profile, task, budget
        splits = profile.predefined_splits
        if splits and "train" in splits:
            self.train_fraction = splits["train"] / profile.n_samples
        else:
            self.train_fraction = 0.7
        self.label_budget = task.label_budget
        if task.label_budget_per_class and profile.labels:
            per_class_train = profile.n_samples * self.train_fraction / profile.labels.n_classes
            self.label_budget = min(1.0, task.label_budget_per_class / per_class_train)
        fraction = self.train_fraction * (self.label_budget or 1.0)
        self.n_train = int(profile.n_samples * fraction)
        if budget.max_train_samples:
            self.n_train = min(self.n_train, budget.max_train_samples)
        self.class_train = {}
        if profile.labels:
            scale = self.n_train / profile.n_samples
            self.class_train = {k: max(1, int(v * scale)) for k, v in
                                profile.labels.class_counts.items()}
        self.p_enc = profile.n_encoded_features
        self.has_labels = profile.labels is not None
        self.is_mixed = (profile.n_categorical + profile.n_discrete + profile.n_binary) > 0
        self.seconds = (budget.train_seconds_image if profile.modality == "image"
                        else budget.train_seconds_other)


# ---------------------------------------------------------------------------
# Step 1: is generative modeling warranted at all?
# ---------------------------------------------------------------------------

def assess_warrant(ctx: Context, log: DecisionLog) -> tuple[bool, str, str | None, TaskSpec]:
    """Return (use_generative_model, rationale, alternative, possibly-adjusted task)."""
    pr, task = ctx.profile, ctx.task.model_copy(deep=True)
    labels = pr.labels
    ev = {"modality": pr.modality, "n_train_est": ctx.n_train, "p_encoded": ctx.p_enc,
          "task": task.task_type}

    def decline(reason: str, alternative: str):
        log.add("warrant", "use a generative model?", "no", reason,
                [("train a generative model", reason)], ev)
        return False, reason, alternative, task

    def accept(reason: str, rejected: list[tuple[str, str]]):
        log.add("warrant", "use a generative model?", "yes", reason, rejected, ev)
        return True, reason, None, task

    t = task.task_type
    if t == "imputation":
        ev["missing_fraction"] = pr.missing.overall_fraction
        if pr.missing.overall_fraction == 0:
            return decline("The data contain no missing values, so there is nothing to impute.",
                           "No imputation needed; analyse the complete data directly.")
        if pr.modality != "tabular":
            return decline(f"Imputation is supported for tabular data; this is {pr.modality}.",
                           "Use modality-specific completion (e.g. inpainting) outside this agent.")
        return accept(
            f"{100 * pr.missing.overall_fraction:.1f}% of cells are missing across "
            f"{pr.missing.n_features_with_missing} variables and "
            f"{100 * pr.missing.rows_with_any_missing_fraction:.0f}% of rows; complete-case "
            "analysis would discard most rows. A deep generative imputer is compared against "
            "mean/mode, kNN and MICE, and is only recommended if it beats them.",
            [("complete-case analysis", "discards too many rows and biases results unless MCAR"),
             ("single mean imputation only", "kept as a baseline; understates variance")])

    if t == "uncertainty_quantification":
        if pr.missing.overall_fraction > 0 and pr.modality == "tabular":
            task.task_type, task.secondary_tasks = "imputation", ["uncertainty_quantification"]
            return accept("Uncertainty here comes from missing data; a generative imputer "
                          "gives multiple imputations whose spread quantifies it.",
                          [("conformal prediction", "addresses predictive, not missing-data, "
                            "uncertainty")])
        return decline(
            "Predictive uncertainty does not require a generative model of the inputs; "
            "a generative model would add modelling error without a benefit.",
            "Use split-conformal prediction intervals or a bootstrap/deep ensemble around the "
            "predictive model, and check calibration (reliability diagram, ECE).")

    if t == "anomaly_detection":
        if pr.modality == "tabular" and ctx.p_enc > MAX_FLOW_DIM:
            return decline(f"{ctx.p_enc} encoded features is too many for a density model "
                           "within budget.", "Use Isolation Forest or a one-class SVM on PCA "
                           "scores.")
        if labels and labels.minority_count >= 50 and labels.minority_fraction > 0.2:
            return decline(
                f"The 'anomalous' class is {100 * labels.minority_fraction:.0f}% of the data "
                f"(n={labels.minority_count}); that is a supervised classification problem.",
                "Train a supervised classifier with class weights and tune its threshold.")
        return accept(
            "Anomaly scoring needs an estimate of the normal-data density; generative models "
            "with a likelihood provide one directly. Isolation Forest is run as a baseline.",
            [("supervised classifier only", "anomalies are rare "
              f"({'%.1f%%' % (100 * labels.minority_fraction) if labels else 'unlabelled'}); "
              "a density model generalises to unseen anomaly types")])

    if t == "representation_learning":
        if pr.modality == "tabular" and ctx.p_enc <= 10:
            return decline(f"Only {ctx.p_enc} features: the data are already low-dimensional.",
                           "Use the standardised features or PCA directly.")
        return accept("A VAE gives a nonlinear, probabilistic embedding; compared with PCA.",
                      [("PCA only", "kept as the baseline embedding")])

    # --- generation / augmentation / class_balancing ------------------------
    if t in ("class_balancing", "augmentation") and not labels:
        if t == "class_balancing":
            return decline("Class balancing was requested but the data have no labels.",
                           "Provide a label column (--label-column) or cluster first.")
        task.task_type = "generation"
    if task.conditional and not labels:
        task.conditional = False
        log.add("warrant", "conditional generation?", "unconditional",
                "No label column found, so class-conditional generation is impossible.",
                [("conditional", "no labels")], ev)

    if labels and t == "class_balancing" and labels.imbalance_ratio < 1.5 \
            and not task.target_classes:
        task.task_type = "augmentation"
        log.add("warrant", "interpretation of 'rebalance'", "augmentation",
                f"Largest/smallest class ratio is only {labels.imbalance_ratio:.2f}; there is "
                "no meaningful imbalance, so the request is treated as augmentation.",
                [("class_balancing", "classes already balanced")],
                {"imbalance_ratio": labels.imbalance_ratio})

    if (pr.modality == "tabular" and labels and task.downstream_task == "classification"
            and task.task_type in ("augmentation", "class_balancing")
            and labels.imbalance_ratio < MILD_IMBALANCE and ctx.p_enc <= LOW_DIM_TABULAR
            and not ctx.label_budget):
        ev.update(imbalance_ratio=labels.imbalance_ratio, minority_count=labels.minority_count)
        return decline(
            f"Low-dimensional tabular data (p={ctx.p_enc}) with mild imbalance "
            f"(ratio {labels.imbalance_ratio:.2f}, minority n={labels.minority_count}). "
            "A generator fitted to these same rows cannot add information about P(y|x) that "
            "a regularised classifier lacks; synthetic augmentation would mostly add "
            "generator bias and give an optimistic picture of performance.",
            "Fit penalised logistic regression and gradient boosting with "
            "class_weight='balanced', select by repeated stratified CV, report calibrated "
            "probabilities, and tune the decision threshold if minority recall matters.")

    if labels and task.task_type == "class_balancing":
        too_small = [c for c, k in ctx.class_train.items() if k < MIN_CLASS_TRAIN]
        targets = task.target_classes or rare_classes(ctx)
        kept = [c for c in targets if c not in too_small]
        if not kept:
            return decline(
                f"Every target class has fewer than {MIN_CLASS_TRAIN} training examples "
                f"({', '.join(targets)}); any generator would reproduce them.",
                "Collect more examples, or merge the rare classes, or use class weights.")
        for c in set(targets) - set(kept):
            log.add("warrant", f"synthesise class '{c}'?", "no",
                    f"~{ctx.class_train[c]} training examples is below the floor of "
                    f"{MIN_CLASS_TRAIN}; synthetic samples would be near-copies.",
                    [("synthesise anyway", "memorisation near-certain")])
        task.target_classes = kept

    return accept(
        f"Objective '{task.task_type}' requires new samples and the data support it "
        f"({pr.brief()}).",
        [("no generative model", "objective explicitly needs synthetic samples; "
          "utility against a real-only baseline will test whether they help")])


def rare_classes(ctx: Context) -> list[str]:
    """Classes with fewer than half the median class count (else the smallest)."""
    counts = ctx.class_train
    if not counts:
        return []
    median = float(np.median(list(counts.values())))
    rare = [c for c, k in counts.items() if k < 0.5 * median]
    return rare or [min(counts, key=counts.get)]


# ---------------------------------------------------------------------------
# Step 2: model choice
# ---------------------------------------------------------------------------

def choose_models(ctx: Context, task: TaskSpec, log: DecisionLog) -> list[ModelSpec]:
    pr = ctx.profile
    t = task.task_type
    cond = task.conditional and ctx.has_labels
    ev = {"modality": pr.modality, "n_train_est": ctx.n_train, "p_encoded": ctx.p_enc}
    models: list[ModelSpec] = []
    rejected: list[tuple[str, str]] = []

    def add(name, role, rep, why, conditional=cond):
        models.append(ModelSpec(name=name, role=role, representation=rep,
                                conditional=conditional, rationale=why))

    if t == "imputation":
        add("masked_vae", "primary", "standardized_onehot",
            "MIWAE-style VAE: its likelihood is evaluated on observed entries only, so it "
            "trains without imputing first, and it returns multiple stochastic imputations.",
            conditional=False)
        rejected += [("ctgan / wgan_gp", "adversarial models need complete rows to train"),
                     ("flows", "need complete rows; no native handling of a missingness mask")]

    elif t == "anomaly_detection":
        if pr.modality == "tabular":
            add("maf", "primary", "standardized",
                "A masked autoregressive flow gives an exact log-density; -log p(x) is a "
                "principled anomaly score, and density evaluation is a single pass.",
                conditional=False)
            add("vae", "comparison", "standardized",
                "VAE negative ELBO as an alternative score with a different inductive bias.",
                conditional=False)
            rejected += [("realnvp", "MAF is the stronger density estimator on tabular "
                          "benchmarks (Papamakarios et al. 2017); one flow suffices"),
                         ("gan", "no likelihood, so no direct anomaly score")]
        else:
            add("vae", "primary", "pixels", "VAE negative ELBO as an anomaly score.",
                conditional=False)
            rejected += [("flows", "pixel-space flows are too costly for the budget")]

    elif t == "representation_learning":
        if pr.modality == "counts":
            name = _count_likelihood(pr, log)
            add(name, "primary", "counts_hvg", "Count-likelihood VAE latent space.",
                conditional=False)
        else:
            add("vae", "primary", "pixels" if pr.modality == "image" else "standardized",
                "VAE encoder mean as the embedding.", conditional=False)

    elif pr.modality == "counts":
        name = _count_likelihood(pr, log)
        add(name, "primary", "counts_hvg",
            "UMI counts are overdispersed integers; a count likelihood with the observed "
            "library size as offset models them directly (scVI-style).")
        add("vae", "comparison", "lognorm_hvg",
            "Gaussian VAE on log-normalised expression: tests whether the count likelihood "
            "actually matters for these data.")
        rejected += [("wgan_gp", f"no likelihood to monitor and prone to mode collapse; with "
                      f"rare classes of ~{min(ctx.class_train.values(), default=0)} cells a "
                      "collapsed mode would erase them"),
                     ("ddpm_mlp", "models continuous transformed values rather than counts and "
                      "is data-hungry at n of a few thousand"),
                     ("flows", f"p={pr.n_features} genes far exceeds the {MAX_FLOW_DIM}-d limit "
                      "for full-rank flows")]

    elif pr.modality == "image":
        small = ctx.n_train < SMALL_IMAGE_TRAIN
        ev["small_image_threshold"] = SMALL_IMAGE_TRAIN
        if small:
            add("vae", "primary", "pixels",
                f"Only ~{ctx.n_train} training images: a convolutional VAE has few parameters, "
                "trains stably, and is less prone to memorising than diffusion.")
            add("ddpm_unet", "comparison", "pixels",
                "Small diffusion model as a comparison; its memorisation check decides "
                "whether it is usable at this n.")
        else:
            add("ddpm_unet", "primary", "pixels",
                f"~{ctx.n_train} training images is enough for a small DDPM, which gives the "
                "best sample fidelity and diversity of the library; classifier-free guidance "
                "conditions it on class.")
            add("vae", "comparison", "pixels",
                "Conditional VAE: cheap, stable comparison; typically blurrier samples.")
        rejected += [("wgan_gp", "adversarial training is unstable at this scale and "
                      "mode collapse would hit minority classes first"),
                     ("realnvp / maf", f"pixel-space flows (p={pr.n_features}) need deep "
                      "multi-scale architectures beyond the budget")]

    else:  # tabular generation / augmentation / balancing
        if (ctx.is_mixed or (pr.shape.fraction_multimodal or 0) >= 0.3) \
                and ctx.n_train < SMALL_TABLE_TRAIN:
            add("vae", "primary", "standardized",
                f"Mixed-type table with only ~{ctx.n_train} training rows: a VAE with a "
                "Gaussian/categorical likelihood (TVAE-style) converges in far fewer updates "
                "than adversarial training and has a validation loss for early stopping.")
            add("ctgan", "comparison", "mode_specific",
                "CTGAN as comparison; at this n it is expected to under-train.")
            rejected += [("ctgan as primary", f"needs many more updates than n~{ctx.n_train} "
                          "rows give within budget (observed in development: continuous "
                          "marginals still biased after 760 epochs on a 400-row table)")]
        elif ctx.is_mixed or (pr.shape.fraction_multimodal or 0) >= 0.3:
            add("ctgan", "primary", "mode_specific",
                f"Mixed column types ({pr.n_categorical} categorical, {pr.n_discrete} discrete, "
                f"{pr.n_binary} binary) and multimodal continuous columns "
                f"({100 * (pr.shape.fraction_multimodal or 0):.0f}%) are what CTGAN's "
                "mode-specific normalisation and conditional sampling were designed for.")
            add("ddpm_mlp", "comparison", "standardized",
                "Tabular diffusion (TabDDPM-style) as a non-adversarial comparison.")
            rejected += [("flows", "continuous-only; categorical columns would need "
                          "dequantisation")]
        elif ctx.p_enc <= MAX_FLOW_DIM and ctx.n_train < 1000:
            add("maf", "primary", "standardized",
                f"All-continuous, n~{ctx.n_train}: a flow has an exact likelihood for model "
                "checking and few hyperparameters to tune at small n.", conditional=False)
            add("ddpm_mlp", "comparison", "standardized", "Tabular diffusion comparison.")
        else:
            add("ddpm_mlp", "primary", "standardized",
                "All-continuous table with enough rows for an MLP diffusion model.")
            if ctx.p_enc <= MAX_FLOW_DIM:
                add("maf", "comparison", "standardized", "Exact-likelihood comparison.",
                    conditional=False)

    primary = models[0]
    log.add("model", "primary model", primary.name, primary.rationale, rejected, ev)
    for m in models[1:]:
        log.add("model", "comparison model", m.name, m.rationale, [], ev)
    return models


def _count_likelihood(pr: DataProfile, log: DecisionLog) -> str:
    c = pr.counts
    pi = c.median_zero_inflation_pi if c else None
    frac = c.fraction_genes_zinb_preferred if c else None
    zinb = (pi is not None and pi > 0.1) or (frac is not None and frac > 0.3)
    ev = {"median_zero_inflation_pi": pi, "fraction_genes_zinb_preferred": frac,
          "median_dispersion_ratio": c.median_dispersion_ratio if c else None}
    if zinb:
        log.add("model", "count likelihood", "ZINB",
                f"Per-gene MLE prefers ZINB for {100 * frac:.0f}% of genes (median fitted "
                f"zero-inflation {pi:.2f}): more zeros than an NB can explain.",
                [("NB", "under-predicts zeros on these data")], ev)
        return "zinb_vae"
    log.add("model", "count likelihood", "NB",
            f"Genes are overdispersed (median var/mean {c.median_dispersion_ratio:.2f}) but "
            f"ZINB beats NB by AIC for only {100 * (frac or 0):.0f}% of genes (median fitted "
            f"zero-inflation {pi if pi is not None else float('nan'):.3f}); the extra "
            "parameter is not supported.",
            [("ZINB", "zero-inflation not supported by per-gene likelihood comparison"),
             ("Poisson", "ignores overdispersion"),
             ("Gaussian on log counts", "misstates the mean-variance relation; kept as the "
              "comparison model")], ev)
    return "nb_vae"


# ---------------------------------------------------------------------------
# Step 3: hyperparameters
# ---------------------------------------------------------------------------

def choose_hyperparameters(ctx: Context, models: list[ModelSpec], log: DecisionLog) -> None:
    pr = ctx.profile
    n = max(ctx.n_train, 1)
    batch = _pow2(n / 20, 32, 256)
    for m in models:
        hp: dict = {"batch_size": batch}
        why: list[str] = [f"batch_size={batch} (~n/20, power of two in [32, 256])"]
        m.train_seconds = ctx.seconds if m.role == "primary" else 0.5 * ctx.seconds
        if m.name in ("nb_vae", "zinb_vae"):
            # scVI's own heuristic: ~400 epochs at n=20k, scaled up for small n, capped.
            epochs = int(min(400, round(20_000 / n * 400)))
            latent = 10 if n >= 1000 else 5
            hp.update(latent_dim=latent, hidden_dim=128, n_layers=1, max_epochs=epochs, lr=1e-3)
            why += [f"latent_dim={latent} (scVI default; 5 if n<1000)",
                    f"max_epochs={epochs} (scVI rule min(400, 20000/n*400))"]
        elif m.name == "vae":
            if pr.modality == "image":
                latent = _pow2((pr.pca_components_90 or 64) / 4, 8, 64)
                why.append(f"latent_dim={latent} (~PCA-90% components {pr.pca_components_90}/4)")
            elif pr.modality == "counts":
                latent = 10
                why.append("latent_dim=10, matched to the count model for a fair comparison")
            else:
                latent = int(min(16, max(2, ctx.p_enc // 2)))
                why.append(f"latent_dim={latent} (p_encoded/2, in [2, 16])")
            hidden = 256 if n >= 2000 else 128
            hp.update(latent_dim=latent, hidden_dim=hidden, beta=1.0, lr=1e-3,
                      weight_decay=0.0 if n >= 2000 else 1e-5)
            why.append(f"hidden_dim={hidden}; weight decay only when n<2000")
        elif m.name == "ddpm_unet":
            ch = 64 if n >= SMALL_IMAGE_TRAIN else 32
            hp.update(base_channels=ch, timesteps=1000, sample_steps=100,
                      guidance_scale=2.0 if m.conditional else 0.0, p_uncond=0.1, lr=2e-4)
            why += [f"base_channels={ch} ({'n>=' if ch == 64 else 'n<'}{SMALL_IMAGE_TRAIN})",
                    "T=1000 training steps, 100 strided sampling steps (cost/quality trade-off)",
                    "guidance 2.0 with 10% label dropout (classifier-free guidance)"
                    if m.conditional else "unconditional"]
        elif m.name == "ddpm_mlp":
            hidden = 512 if n >= 5000 else 256
            hp.update(hidden_dim=hidden, n_layers=4, timesteps=1000, sample_steps=100, lr=1e-3,
                      guidance_scale=1.0 if m.conditional else 0.0)
            why.append(f"hidden_dim={hidden} (512 if n>=5000)")
        elif m.name in ("maf", "realnvp"):
            blocks = 5 if n >= 2000 else 3
            hidden = int(min(256, max(32, 8 * ctx.p_enc)))
            dup = max(pr.duplicate_fraction or 0, pr.near_duplicate_fraction or 0)
            noise = DEQUANTISE_STD if dup > DUPLICATE_TOLERANCE else 0.0
            hp.update(n_blocks=blocks, hidden_dim=hidden, lr=1e-3, noise_std=noise,
                      weight_decay=1e-5 if n >= 2000 else 1e-4)
            why.append(f"n_blocks={blocks}, hidden_dim={hidden} (8 x p, in [32, 256]); "
                       "fewer blocks and more decay at small n")
            if noise:
                why.append(f"noise_std={noise}: {100 * dup:.0f}% of rows are (near-)duplicates, "
                           "on which a continuous density is degenerate, so the flow is "
                           "trained on dequantised (jittered) data")
        elif m.name == "masked_vae":
            latent = int(min(16, max(2, ctx.p_enc // 4)))
            hidden = 64 if n < 1000 else 128
            hp.update(latent_dim=latent, hidden_dim=hidden, n_importance=10, lr=1e-3,
                      weight_decay=1e-4 if n < 1000 else 1e-5)
            why.append(f"latent_dim={latent} (p_encoded/4), hidden_dim={hidden}, "
                       "K=10 importance samples (MIWAE)")
        elif m.name == "ctgan":
            hp.update(latent_dim=128, hidden_dim=256, gp_lambda=10.0, pac=1, max_modes=5, lr=2e-4)
            why.append("CTGAN paper defaults (z=128, 256-wide, lambda_GP=10, <=5 modes/column)")
        elif m.name == "wgan_gp":
            hp.update(gp_lambda=10.0, n_critic=5, lr=1e-4)
            why.append("WGAN-GP paper defaults (lambda=10, 5 critic steps)")
        hp.setdefault("max_epochs", 2000)
        m.hyperparameters = hp
        log.add("hyperparameters", f"{m.name} ({m.role})",
                ", ".join(f"{k}={v}" for k, v in hp.items()),
                "; ".join(why) + f". Training capped at {m.train_seconds:.0f}s wall-clock.",
                [], {"n_train_est": ctx.n_train, "pca_components_90": pr.pca_components_90,
                     "train_seconds": m.train_seconds})


# ---------------------------------------------------------------------------
# Step 4: sampling and evaluation
# ---------------------------------------------------------------------------

def choose_sampling(ctx: Context, task: TaskSpec, log: DecisionLog) -> SamplingPlan:
    t = task.task_type
    n_eval = int(min(ctx.n_train, ctx.budget.n_synthetic_max))
    if t == "class_balancing" and ctx.class_train:
        counts = ctx.class_train
        targets = task.target_classes or rare_classes(ctx)
        others = [k for c, k in counts.items() if c not in targets] or list(counts.values())
        level = int(np.median(others))
        per = {c: int(max(0, min(level, MAX_SYNTH_PER_REAL * counts[c]) - counts[c]))
               for c in targets}
        plan = SamplingPlan(strategy="rebalance_targets", target_classes=targets,
                            n_per_class=per, n_total=n_eval)
        why = (f"Raise each target class to min(median size of the other classes ~{level}, "
               f"{MAX_SYNTH_PER_REAL}x its real count); beyond that, synthetic points are draws "
               "from a smoothed copy of a few real cells, not new evidence.")
    elif t == "augmentation" and ctx.class_train:
        per = {c: int(AUGMENT_RATIO * k) for c, k in ctx.class_train.items()}
        plan = SamplingPlan(strategy="augment_per_class", target_classes=list(per),
                            n_per_class=per, n_total=n_eval)
        why = (f"1:1 synthetic-to-real per class; larger ratios let generator bias dominate "
               "the training distribution.")
    elif t in ("generation", "augmentation", "class_balancing"):
        plan = SamplingPlan(strategy="match_real", n_total=n_eval)
        why = "Sample as many points as the real training set (capped), matching its classes."
    else:
        plan = SamplingPlan(strategy="none", n_total=0)
        why = f"Task '{t}' uses the model's density/posterior, not a synthetic dataset."
    log.add("sampling", "how many samples, of what", plan.strategy, why +
            f" Evaluation sample n={plan.n_total}.", [], {"n_per_class": plan.n_per_class})
    return plan


def choose_evaluation(ctx: Context, task: TaskSpec, log: DecisionLog) -> EvalPlan:
    pr, t = ctx.profile, task.task_type
    space = {"image": "domain_cnn_features", "counts": "lognorm_pca50",
             "tabular": "standardized_onehot"}[pr.modality]
    if t == "imputation":
        ev = EvalPlan(metrics=["masked_rmse", "masked_categorical_accuracy",
                               "multiple_imputation_coverage"],
                      primary_metric="masked_rmse",
                      baselines=["mean_mode", "knn", "mice"], feature_space=space)
        why = ("Hide 20% of observed entries, impute, and score against the hidden truth "
               "(RMSE on standardised continuous variables, accuracy on categoricals); "
               "the deep imputer must beat mean/mode, kNN and MICE to be recommended.")
    elif t == "anomaly_detection":
        ev = EvalPlan(metrics=["auroc", "auprc", "score_calibration"], primary_metric="auprc",
                      baselines=["isolation_forest"], feature_space=space)
        why = ("AUROC and AUPRC on held-out data with known labels; AUPRC is primary "
               "because with rare anomalies AUROC is dominated by the easy negatives.")
    elif t == "representation_learning":
        ev = EvalPlan(metrics=["linear_probe", "knn_preservation"], primary_metric="linear_probe",
                      baselines=["pca"], feature_space=space)
        why = "Embedding quality via a linear probe on labels and kNN preservation, vs PCA."
    else:
        metrics = ["precision_recall", "density_coverage", "frechet_distance", "mmd",
                   "memorization"]
        if pr.modality != "image":
            metrics += ["marginals", "correlation"]
        if ctx.has_labels:
            metrics += ["utility_augmented"]
            if task.wants_tstr or t in ("generation", "augmentation"):
                metrics.append("utility_tstr")
            if t == "class_balancing":
                metrics.append("per_class_recall")
        primary = {"class_balancing": "target_class_recall_gain",
                   "augmentation": "macro_f1_gain"}.get(t, "frechet_distance")
        if not ctx.has_labels:
            primary = "frechet_distance"
        ev = EvalPlan(metrics=metrics, primary_metric=primary,
                      baselines=["real_only_classifier"] if ctx.has_labels else [],
                      feature_space=space)
        why = (f"Fidelity, diversity and memorization measured in {space} (a domain-appropriate "
               "space: Inception features would be mis-specified for these data); utility via a "
               f"downstream classifier. Primary metric '{primary}' is the quantity the "
               "objective asks to improve.")
    log.add("evaluation", "evaluation criteria", ", ".join(ev.metrics), why,
            [("Inception-based FID", "ImageNet features do not describe this domain")]
            if pr.modality == "image" else [], {"primary_metric": ev.primary_metric})
    return ev


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plan(profile: DataProfile, task: TaskSpec, budget: Budget, log: DecisionLog,
         source: str = "heuristic") -> Plan:
    ctx = Context(profile, task, budget)
    use, rationale, alternative, task = assess_warrant(ctx, log)
    ctx.task = task
    data = DataPlan(use_predefined_splits=bool(profile.predefined_splits),
                    label_budget=ctx.label_budget, max_train_samples=budget.max_train_samples)
    if profile.modality == "counts":
        assessed = profile.counts.n_features_assessed if profile.counts else profile.n_features
        data.n_hvg = int(min(2000, max(200, assessed)))
        log.add("preprocessing", "genes modelled", f"top {data.n_hvg} highly variable genes",
                f"{assessed} genes are detected in >=5% of cells; 2,000 HVGs is the standard "
                "Scanpy/Seurat choice and keeps p comparable to n.",
                [("all genes", f"p={profile.n_features} >> n, mostly near-silent")],
                {"n_features_assessed": assessed})
    if ctx.label_budget:
        log.add("preprocessing", "labelled training subset",
                f"{100 * ctx.label_budget:.1f}% stratified subsample of train",
                "The objective states that only this share of labels is available; the "
                "generator and every classifier see only this subset.",
                [], {"label_budget": ctx.label_budget, "n_train_est": ctx.n_train})

    if not use:
        return Plan(use_generative_model=False, decision_rationale=rationale,
                    alternative_if_not=alternative, task=task, data=data, source=source)

    models = choose_models(ctx, task, log)
    choose_hyperparameters(ctx, models, log)
    return Plan(
        use_generative_model=True, decision_rationale=rationale, task=task, data=data,
        models=models, sampling=choose_sampling(ctx, task, log),
        evaluation=choose_evaluation(ctx, task, log), source=source,
    )
