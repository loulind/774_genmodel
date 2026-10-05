"""Check a plan against the guardrails; repair what can be repaired.

Two layers, as in Project 1. Pydantic rejects anything malformed (wrong types,
unknown task names). This module then enforces what a type system cannot:

  * models must exist in the registry and suit the modality and the task;
  * conditional models need labels; flows need modest dimension;
  * hyperparameters must lie in registry ranges (clipped, with a note);
  * compute: per-model wall-clock caps and a total budget;
  * sampling sizes within caps; named target classes must exist;
  * a "do not generate" plan must give a reason and an alternative.

Repairs are recorded and logged. A plan whose *primary* model is unusable, or
that fails schema validation outright, is not repaired: the caller falls back
to the heuristic planner, and the decision log says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from ..config import Budget
from ..models.registry import MAX_FLOW_DIM, MAX_MODELS_PER_PLAN, REGISTRY, clip_hyperparameters
from ..schema import DataProfile, Plan

KNOWN_METRICS = {
    "precision_recall", "density_coverage", "frechet_distance", "mmd", "memorization",
    "marginals", "correlation", "utility_augmented", "utility_tstr", "per_class_recall",
    "masked_rmse", "masked_categorical_accuracy", "multiple_imputation_coverage",
    "auroc", "auprc", "score_calibration", "linear_probe", "knn_preservation",
}
KNOWN_PRIMARY = KNOWN_METRICS | {"target_class_recall_gain", "macro_f1_gain"}
DEFAULT_REPRESENTATION = {
    ("image", "vae"): "pixels", ("image", "wgan_gp"): "pixels", ("image", "ddpm_unet"): "pixels",
    ("counts", "nb_vae"): "counts_hvg", ("counts", "zinb_vae"): "counts_hvg",
    ("counts", "vae"): "lognorm_hvg", ("counts", "wgan_gp"): "lognorm_hvg",
    ("counts", "ddpm_mlp"): "lognorm_pca",
    ("tabular", "ctgan"): "mode_specific", ("tabular", "masked_vae"): "standardized_onehot",
}
# Total training wall-clock across all models, as a multiple of the per-model cap.
TOTAL_BUDGET_MULTIPLE = 2.0


@dataclass
class ValidationResult:
    plan: Plan | None
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.errors


def validate(candidate: dict | Plan, profile: DataProfile, budget: Budget) -> ValidationResult:
    try:
        plan = candidate if isinstance(candidate, Plan) else Plan(**candidate)
    except ValidationError as err:
        return ValidationResult(None, errors=[f"schema: {e['loc']}: {e['msg']}"
                                              for e in err.errors()][:10])
    plan = plan.model_copy(deep=True)
    errors, repairs = [], []
    modality = profile.modality
    labels = profile.labels

    if not plan.use_generative_model:
        if not plan.alternative_if_not or not plan.decision_rationale:
            errors.append("a 'do not generate' plan must give a rationale and an alternative")
        if plan.models:
            repairs.append("removed models from a 'do not generate' plan")
            plan.models = []
        return ValidationResult(plan if not errors else None, errors, repairs)

    if not plan.models:
        return ValidationResult(None, ["plan uses a generative model but lists none"])

    task = plan.task.task_type
    kept = []
    for m in plan.models:
        info = REGISTRY.get(m.name)
        problem = None
        if info is None:
            problem = f"unknown model '{m.name}'; allowed: {sorted(REGISTRY)}"
        elif modality not in info.modalities:
            problem = f"{m.name} does not support modality '{modality}'"
        elif task not in info.tasks:
            problem = f"{m.name} is not suited to task '{task}'"
        elif info.family == "flow" and profile.n_encoded_features > MAX_FLOW_DIM:
            problem = (f"{m.name} needs <= {MAX_FLOW_DIM} dimensions; data have "
                       f"{profile.n_encoded_features}")
        if problem:
            if m.role == "primary":
                errors.append(f"primary model invalid: {problem}")
            else:
                repairs.append(f"dropped comparison model: {problem}")
            continue

        if m.conditional and not labels:
            m.conditional = False
            repairs.append(f"{m.name}: conditional=False (no labels)")
        if m.conditional and not info.conditional_capable:
            m.conditional = False
            repairs.append(f"{m.name}: conditional=False (model not conditional-capable)")
        if m.representation not in info.representations:
            new = DEFAULT_REPRESENTATION.get((modality, m.name), info.representations[0])
            repairs.append(f"{m.name}: representation '{m.representation}' -> '{new}'")
            m.representation = new
        m.hyperparameters, notes = clip_hyperparameters(m.name, m.hyperparameters)
        repairs += notes
        cap = budget.train_seconds_image if modality == "image" else budget.train_seconds_other
        if m.train_seconds > cap or m.train_seconds <= 0:
            repairs.append(f"{m.name}: train_seconds {m.train_seconds:.0f} -> {cap:.0f} (cap)")
            m.train_seconds = cap
        kept.append(m)

    if errors:
        return ValidationResult(None, errors, repairs)

    primaries = [m for m in kept if m.role == "primary"]
    if not primaries:
        kept[0].role = "primary"
        repairs.append(f"promoted {kept[0].name} to primary")
    for extra in primaries[1:]:
        extra.role = "comparison"
        repairs.append(f"demoted {extra.name} to comparison (one primary allowed)")
    kept.sort(key=lambda m: m.role != "primary")
    if len(kept) > MAX_MODELS_PER_PLAN:
        repairs.append(f"kept {MAX_MODELS_PER_PLAN} of {len(kept)} models")
        kept = kept[:MAX_MODELS_PER_PLAN]

    cap = budget.train_seconds_image if modality == "image" else budget.train_seconds_other
    total = sum(m.train_seconds for m in kept)
    if total > TOTAL_BUDGET_MULTIPLE * cap:
        room = TOTAL_BUDGET_MULTIPLE * cap - kept[0].train_seconds
        others = sum(m.train_seconds for m in kept[1:])
        for m in kept[1:]:
            m.train_seconds = max(10.0, m.train_seconds * room / others)
        repairs.append(f"scaled comparison-model time so total <= {TOTAL_BUDGET_MULTIPLE}x cap")
    plan.models = kept

    # sampling
    s = plan.sampling
    if labels:
        unknown = [c for c in s.target_classes if c not in labels.class_names]
        if unknown:
            s.target_classes = [c for c in s.target_classes if c in labels.class_names]
            s.n_per_class = {c: k for c, k in s.n_per_class.items() if c in labels.class_names}
            repairs.append(f"dropped unknown target classes {unknown}")
    if s.n_total > budget.n_synthetic_max:
        repairs.append(f"n_total {s.n_total} -> {budget.n_synthetic_max}")
        s.n_total = budget.n_synthetic_max
    per_total = sum(s.n_per_class.values())
    if per_total > budget.n_synthetic_max:
        scale = budget.n_synthetic_max / per_total
        s.n_per_class = {c: int(k * scale) for c, k in s.n_per_class.items()}
        repairs.append(f"scaled per-class synthetic counts by {scale:.2f} to fit the cap")

    # data plan
    if plan.data.label_budget is not None and not 0 < plan.data.label_budget <= 1:
        repairs.append(f"label_budget {plan.data.label_budget} invalid -> None")
        plan.data.label_budget = None

    # evaluation
    if plan.evaluation is None:
        errors.append("plan has no evaluation section")
    else:
        bad = [x for x in plan.evaluation.metrics if x not in KNOWN_METRICS]
        if bad:
            plan.evaluation.metrics = [x for x in plan.evaluation.metrics if x in KNOWN_METRICS]
            repairs.append(f"dropped unknown metrics {bad}")
        if plan.evaluation.primary_metric not in KNOWN_PRIMARY:
            new = plan.evaluation.metrics[0] if plan.evaluation.metrics else "frechet_distance"
            repairs.append(f"primary metric '{plan.evaluation.primary_metric}' -> '{new}'")
            plan.evaluation.primary_metric = new

    return ValidationResult(None if errors else plan, errors, repairs)
