"""Objective-specific evaluations: imputation, anomaly detection, representation.

Imputation. We hide 20% of the *observed* entries of the test rows, impute them,
and score against the hidden truth: RMSE on standardised continuous variables
(so variables on different scales count equally) and accuracy on categoricals.
Baselines: mean/mode, k-nearest-neighbours (k=5) and MICE (iterative chained
equations, van Buuren & Groothuis-Oudshoorn 2011), all fitted on the training
rows. For multiple imputation we also report the coverage of nominal 90%
intervals for the hidden continuous values -- proper multiple imputation should
cover about 90%; single imputation has no interval at all.

Anomaly detection. Scores are -log-likelihood (or bound). AUROC, AUPRC and
precision at the prevalence (the share of true anomalies among the top-k scores,
k = number of anomalies) against an Isolation Forest baseline (Liu et al. 2008).

Representation. Linear-probe accuracy from the latent means vs PCA with the same
number of dimensions, and 10-NN preservation between data and latent space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

HOLDOUT_FRACTION = 0.2
N_DRAWS = 20


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def _with_nan(Z, M):
    out = Z.astype(float).copy()
    out[~M.astype(bool)] = np.nan
    return out


def _harden(Z, spans):
    Z = Z.copy()
    for a, b, kind in spans:
        if kind == "cat":
            Z[:, a:b] = np.eye(b - a)[np.nan_to_num(Z[:, a:b], nan=-1).argmax(1)]
    return Z


def imputation_eval(rep, model, train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    spans = rep.spans
    Ztr, Mtr = rep.transform(train_df), rep.observed_mask(train_df).astype(bool)
    Zte, Mte = rep.transform(test_df), rep.observed_mask(test_df).astype(bool)

    # Hide 20% of observed *variables* per test row (a whole one-hot block at once).
    hide = np.zeros_like(Mte)
    for a, b, kind in spans:
        observed = Mte[:, a]
        chosen = observed & (rng.random(len(Zte)) < HOLDOUT_FRACTION)
        hide[chosen, a:b] = True
    M_eval = Mte & ~hide
    cont = [a for a, b, k in spans if k == "cont"]
    cats = [(a, b) for a, b, k in spans if k == "cat"]

    def score(Zhat):
        out = {}
        if cont:
            h = hide[:, cont]
            diff = (Zhat[:, cont] - Zte[:, cont])[h]
            out["masked_rmse"] = float(np.sqrt(np.mean(diff ** 2))) if h.any() else None
        hits, total = 0, 0
        for a, b in cats:
            rows = hide[:, a]
            hits += int((Zhat[rows, a:b].argmax(1) == Zte[rows, a:b].argmax(1)).sum())
            total += int(rows.sum())
        out["masked_categorical_accuracy"] = hits / total if total else None
        return out

    results = {}
    Xtr_nan, Xte_nan = _with_nan(Ztr, Mtr), _with_nan(Zte, M_eval)
    baselines = {
        "mean_mode": SimpleImputer(strategy="mean"),
        "knn": KNNImputer(n_neighbors=5),
        "mice": IterativeImputer(max_iter=10, random_state=seed),
    }
    for name, imp in baselines.items():
        imp.fit(Xtr_nan)
        results[name] = score(_harden(imp.transform(Xte_nan), spans))

    results[model.spec.name] = score(model.impute(Zte, M_eval.astype(np.float32)))

    # Multiple-imputation interval coverage (continuous hidden entries).
    if cont:
        h = hide[:, cont]
        truth = Zte[:, cont][h]
        draws = model.impute(Zte, M_eval.astype(np.float32), n_draws=N_DRAWS)[:, :, cont][:, h]
        lo, hi = np.quantile(draws, 0.05, 0), np.quantile(draws, 0.95, 0)
        results[model.spec.name]["multiple_imputation_coverage"] = float(
            ((truth >= lo) & (truth <= hi)).mean())
        mice_draws = []
        for k in range(N_DRAWS):
            imp = IterativeImputer(max_iter=10, sample_posterior=True, random_state=seed + k)
            mice_draws.append(imp.fit(Xtr_nan).transform(Xte_nan)[:, cont][h])
        mice_draws = np.array(mice_draws)
        lo, hi = np.quantile(mice_draws, 0.05, 0), np.quantile(mice_draws, 0.95, 0)
        results["mice"]["multiple_imputation_coverage"] = float(((truth >= lo) & (truth <= hi)).mean())

    best_baseline = min((r["masked_rmse"], n) for n, r in results.items()
                        if n in baselines and r.get("masked_rmse") is not None)
    return {"methods": results, "n_hidden_cells": int(hide[:, [a for a, *_ in spans]].sum()),
            "best_baseline": best_baseline[1],
            "model_beats_best_baseline": bool(
                results[model.spec.name].get("masked_rmse") is not None
                and results[model.spec.name]["masked_rmse"] < best_baseline[0])}


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _anomaly_metrics(y, s) -> dict:
    k = int(y.sum())
    top = np.argsort(-s)[:k]
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "precision_at_prevalence": float(y[top].mean()) if k else None}


def anomaly_eval(scores_by_model: dict, X_train_normal: np.ndarray, X_test: np.ndarray,
                 y_test_anomaly: np.ndarray, seed: int) -> dict:
    out = {"prevalence": float(y_test_anomaly.mean()), "n_test": int(len(y_test_anomaly)),
           "methods": {}}
    for name, s in scores_by_model.items():
        out["methods"][name] = _anomaly_metrics(y_test_anomaly, s)
    iso = IsolationForest(random_state=seed).fit(X_train_normal)
    out["methods"]["isolation_forest"] = _anomaly_metrics(y_test_anomaly,
                                                          -iso.score_samples(X_test))
    return out


# ---------------------------------------------------------------------------
# Representation learning
# ---------------------------------------------------------------------------

def representation_eval(Z_train, Z_test, X_train, X_test, y_train, y_test, seed) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    k = Z_train.shape[1]
    pca = PCA(k, random_state=seed).fit(X_train)
    P_train, P_test = pca.transform(X_train), pca.transform(X_test)

    def probe(A, B):
        if y_train is None:
            return None
        clf = LogisticRegression(max_iter=3000).fit(A, y_train)
        return float((clf.predict(B) == y_test).mean())

    def knn_overlap(A, B, k_nn=10):
        ia = NearestNeighbors(n_neighbors=k_nn + 1).fit(A).kneighbors(A)[1][:, 1:]
        ib = NearestNeighbors(n_neighbors=k_nn + 1).fit(B).kneighbors(B)[1][:, 1:]
        return float(np.mean([len(set(a) & set(b)) / k_nn for a, b in zip(ia, ib)]))

    return {"latent": {"linear_probe": probe(Z_train, Z_test),
                       "knn_preservation": knn_overlap(X_test, Z_test)},
            "pca": {"linear_probe": probe(P_train, P_test),
                    "knn_preservation": knn_overlap(X_test, P_test)},
            "dimensions": int(k)}
