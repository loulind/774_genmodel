"""Evaluation metrics behave as their definitions say on constructed cases."""

import numpy as np
import pandas as pd

from genmodel_agent.data.splits import Subset
from genmodel_agent.evaluate import evaluate_generative, fidelity, privacy
from genmodel_agent.evaluate.features import EvalSpace
from genmodel_agent.evaluate.task import anomaly_eval, imputation_eval
from genmodel_agent.preprocess import build_representation
from genmodel_agent.schema import ModelSpec

rng = np.random.default_rng(3)


def two_clusters(n):
    return np.r_[rng.normal(-3, 1, (n // 2, 4)), rng.normal(3, 1, (n - n // 2, 4))]


def test_precision_recall_detect_collapse_and_shift():
    real = two_clusters(1000)
    good = fidelity.precision_recall_density_coverage(real, two_clusters(1000))
    collapsed = fidelity.precision_recall_density_coverage(real, rng.normal(3, 1, (1000, 4)))
    shifted = fidelity.precision_recall_density_coverage(real, two_clusters(1000) + 4)
    assert good["precision"] > 0.8 and good["recall"] > 0.8
    assert collapsed["precision"] > 0.8 and collapsed["recall"] < 0.6   # fidelity kept, diversity lost
    assert shifted["precision"] < good["precision"] - 0.3


def test_distances_rank_correctly():
    real = two_clusters(800)
    same, moved = two_clusters(800), two_clusters(800) + 1.5
    assert fidelity.frechet_distance(real, same) < fidelity.frechet_distance(real, moved)
    assert fidelity.mmd_rbf(real, same, rng) < fidelity.mmd_rbf(real, moved, rng)


def test_memorization_flag():
    train, holdout = two_clusters(600), two_clusters(600)
    copies = train[rng.integers(0, 600, 600)] + rng.normal(0, 1e-3, (600, 4))
    assert privacy.memorization(train, holdout, copies)["memorization_flag"]
    assert not privacy.memorization(train, holdout, two_clusters(600))["memorization_flag"]


class _MeanImputer:
    spec = ModelSpec(name="stub")

    def impute(self, Z, M, n_draws=1):
        filled = np.where(M > 0, Z, 0.0)          # 0 = training mean after standardising
        return filled if n_draws == 1 else np.stack([filled] * n_draws)


def test_imputation_eval_baselines():
    n = 400
    z = rng.normal(size=n)
    df = pd.DataFrame({"a": z + 0.1 * rng.normal(size=n), "b": 2 * z + 0.1 * rng.normal(size=n),
                       "c": -z + 0.1 * rng.normal(size=n),
                       "g": np.where(z > 0, "pos", "neg")})
    df.loc[rng.random(n) < 0.1, "a"] = np.nan
    rep = build_representation("standardized_onehot").fit(Subset(df[:300], None, np.arange(300)))
    out = imputation_eval(rep, _MeanImputer(), df[:300], df[300:].reset_index(drop=True), 0)
    m = out["methods"]
    assert m["mice"]["masked_rmse"] < 0.5 * m["mean_mode"]["masked_rmse"]
    assert abs(m["stub"]["masked_rmse"] - m["mean_mode"]["masked_rmse"]) < 0.15
    assert out["best_baseline"] in {"mice", "knn"} and not out["model_beats_best_baseline"]
    assert 0.7 < m["mice"]["multiple_imputation_coverage"] <= 1.0


def test_anomaly_eval():
    y = (rng.random(500) < 0.05).astype(int)
    X = rng.normal(size=(500, 3)) + 4 * y[:, None]
    out = anomaly_eval({"oracle": y + 0.01 * rng.normal(size=500)}, X[y == 0], X, y, 0)
    assert out["methods"]["oracle"]["auroc"] > 0.99
    assert out["methods"]["isolation_forest"]["auroc"] > 0.9


def test_evaluate_generative_tabular_end_to_end():
    n = 600
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(rng.normal(size=(n, 3)) + y[:, None], columns=list("xyz"))
    df["k"] = np.where(rng.random(n) < 0.5, "u", "v")
    tr, va, te = Subset(df[:400], y[:400], np.arange(400)), \
        Subset(df[400:450].reset_index(drop=True), y[400:450], np.arange(50)), \
        Subset(df[450:].reset_index(drop=True), y[450:], np.arange(150))
    space = EvalSpace("tabular", "cpu", 0).fit(tr, va)
    ys = rng.integers(0, 2, 400)
    synth = pd.DataFrame(rng.normal(size=(400, 3)) + ys[:, None], columns=list("xyz"))
    synth["k"] = np.where(rng.random(400) < 0.5, "u", "v")
    real = {"train": tr.X, "y_train": tr.y, "val": va.X, "y_val": va.y, "test": te.X, "y_test": te.y}
    res = evaluate_generative(space, real, {"m": {"eval": (synth, ys), "augment": (synth[:100], ys[:100])}},
                              ["precision_recall", "frechet_distance", "mmd", "memorization",
                               "marginals", "correlation", "utility_augmented", "utility_tstr"],
                              targets=[1], class_names=["a", "b"], n_classes=2, seeds=[0],
                              want_tstr=True, max_samples=1000, seed=0)
    m = res["models"]["m"]
    assert m["precision"] > 0.7 and not m["memorization_flag"]
    assert "target_classes" in m and "b" in m["target_classes"]
    u = res["utility"]["models"]["m"]
    assert abs(u["tstr"]["macro_f1"] - res["utility"]["real_only"]["macro_f1"]) < 0.1
    assert u["macro_f1_gain"]["ci_low"] <= u["macro_f1_gain"]["gain"] <= u["macro_f1_gain"]["ci_high"]
