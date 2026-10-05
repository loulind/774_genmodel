"""Objective parsing, heuristic planning and plan validation."""

import ast
from pathlib import Path

import pytest

from genmodel_agent.config import load_config
from genmodel_agent.decision_log import DecisionLog
from genmodel_agent.objective import parse_objective
from genmodel_agent.planner import heuristic
from genmodel_agent.planner.validator import validate
from genmodel_agent.schema import (
    CountSummary, DataProfile, LabelSummary, MissingSummary, ShapeSummary,
)

BUDGET = load_config().budget("mps")
NO_MISSING = MissingSummary(overall_fraction=0, n_features_with_missing=0,
                            max_feature_fraction=0, rows_with_any_missing_fraction=0)


def make_profile(modality="tabular", n=1000, p=10, counts=None, missing=NO_MISSING,
                 classes=None, **kw):
    labels = None
    if classes:
        k = list(classes.values())
        labels = LabelSummary(n_classes=len(k), class_names=list(classes), class_counts=classes,
                              imbalance_ratio=max(k) / min(k), minority_class=min(classes, key=classes.get),
                              minority_count=min(k), minority_fraction=min(k) / sum(k))
    return DataProfile(name="t", source="t", container="csv", modality=modality,
                       modality_evidence="test", n_samples=n, n_features=p,
                       n_encoded_features=p, n_over_p=n / p, labels=labels, missing=missing,
                       counts=counts, shape=ShapeSummary(), dense_memory_mb=1.0,
                       n_continuous=p, pca_components_90=kw.pop("pca90", 20), **kw)


def run(profile, objective):
    log = DecisionLog()
    plan = heuristic.plan(profile, parse_objective(objective, profile), BUDGET, log)
    return plan, log


# --- objective parsing ------------------------------------------------------

def test_parse_label_budget_and_tstr():
    t = parse_objective("Only a small labelled subset (~5% of training patches) is available. "
                        "Generate class-conditional patches; test a classifier trained on "
                        "synthetic data alone.")
    assert t.task_type == "augmentation" and t.label_budget == pytest.approx(0.05)
    assert t.conditional and t.wants_tstr and t.downstream_task == "classification"


@pytest.mark.parametrize("text, task", [
    ("Impute the missing lab values", "imputation"),
    ("Flag anomalous transactions", "anomaly_detection"),
    ("Rebalance the rare classes", "class_balancing"),
    ("Learn a low-dimensional embedding of the cells", "representation_learning"),
    ("Generate synthetic records for sharing", "generation"),
])
def test_parse_task_types(text, task):
    assert parse_objective(text).task_type == task


# --- warrant ---------------------------------------------------------------

def test_declines_low_dim_tabular_mild_imbalance():
    plan, log = run(make_profile(n=569, p=30, classes={"a": 357, "b": 212}),
                    "Generate synthetic patients to augment the data and improve a classifier")
    assert not plan.use_generative_model and plan.alternative_if_not
    assert validate(plan, make_profile(), BUDGET).ok
    assert log.entries[-1].chosen == "no"


def test_declines_imputation_without_missing():
    plan, _ = run(make_profile(), "Impute missing values")
    assert not plan.use_generative_model


def test_excludes_class_below_floor():
    plan, log = run(make_profile(n=2000, p=50, modality="image", image_shape=[5, 10, 1],
                                 classes={"big": 1980, "mid": 1500, "tiny": 5, "small": 40}),
                    "Generate images to rebalance the rare classes")
    assert plan.use_generative_model
    assert plan.task.target_classes == ["small"]          # tiny: ~3 train examples < floor
    assert any("tiny" in e.decision for e in log.entries)


# --- model choice ------------------------------------------------------------

def _counts(pi):
    return CountSummary(mean_library_size=2000, median_library_size=2000,
                        fraction_overdispersed_features=0.9, median_dispersion_ratio=1.5,
                        observed_zero_fraction=0.8, poisson_expected_zero_fraction=0.7,
                        nb_expected_zero_fraction=0.8, excess_zeros_vs_nb=0.0,
                        n_features_assessed=3000, fraction_genes_zinb_preferred=pi,
                        median_zero_inflation_pi=pi)


@pytest.mark.parametrize("pi, model", [(0.01, "nb_vae"), (0.4, "zinb_vae")])
def test_count_likelihood_follows_profile(pi, model):
    plan, _ = run(make_profile("counts", n=3000, p=20000, counts=_counts(pi),
                               classes={"a": 2900, "b": 100}),
                  "Generate synthetic cells for rare cell types")
    assert plan.models[0].name == model and plan.models[0].conditional


@pytest.mark.parametrize("n, primary", [(800, "vae"), (50_000, "ddpm_unet")])
def test_image_model_depends_on_n(n, primary):
    plan, _ = run(make_profile("image", n=n, p=784, image_shape=[28, 28, 1],
                               classes={"a": n // 2, "b": n // 2}),
                  "Generate class-conditional images")
    assert plan.models[0].name == primary


def test_heuristic_plans_pass_validation():
    prof = make_profile(n=5000, p=8, classes={"n": 4900, "a": 100})
    plan, _ = run(prof, "Detect anomalous cases")
    res = validate(plan, prof, BUDGET)
    assert res.ok and not res.errors and plan.models[0].name == "maf"


# --- validator guardrails ----------------------------------------------------

def _plan_dict(models, **kw):
    return {"use_generative_model": True, "decision_rationale": "x",
            "task": {"raw_objective": "x", "task_type": "generation"},
            "models": models, "evaluation": {"metrics": ["mmd"], "primary_metric": "mmd"}, **kw}


def test_validator_rejects_unknown_primary():
    res = validate(_plan_dict([{"name": "transformer", "role": "primary"}]), make_profile(), BUDGET)
    assert not res.ok and "unknown model" in res.errors[0]


def test_validator_repairs():
    prof = make_profile(p=500)
    res = validate(_plan_dict([
        {"name": "ddpm_mlp", "role": "primary", "conditional": True,
         "hyperparameters": {"lr": 5.0, "made_up": 1}, "train_seconds": 1e6},
        {"name": "maf", "role": "comparison"},          # p=500 > flow limit
    ]), prof, BUDGET)
    assert res.ok
    m = res.plan.models
    assert len(m) == 1 and m[0].hyperparameters["lr"] == 1e-2 and not m[0].conditional
    assert "made_up" not in m[0].hyperparameters
    assert m[0].train_seconds == BUDGET.train_seconds_other
    assert any("dropped comparison" in r for r in res.repairs)


def test_validator_requires_alternative_when_declining():
    res = validate({"use_generative_model": False, "decision_rationale": "no",
                    "task": {"raw_objective": "x", "task_type": "generation"}},
                   make_profile(), BUDGET)
    assert not res.ok


def test_schema_garbage_is_rejected():
    assert not validate({"use_generative_model": "maybe"}, make_profile(), BUDGET).ok


# --- hard rule: no dataset-specific branches -----------------------------------

BANNED = ("pbmc", "pathmnist", "breastmnist", "medmnist", "kidney", "mammograph", "wisconsin")


def test_no_dataset_names_in_code():
    """Dataset names may appear in docstrings/comments, never in code or strings."""
    offenders = []
    for path in Path("genmodel_agent").rglob("*.py"):
        tree = ast.parse(path.read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                                           ast.AsyncFunctionDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                text = node.value
            elif isinstance(node, ast.Name):
                text = node.id
            if text and any(b in text.lower() for b in BANNED):
                offenders.append(f"{path}:{node.lineno}: {text[:60]}")
    assert not offenders, offenders
