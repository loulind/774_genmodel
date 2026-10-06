"""Downstream utility: does synthetic data help a classifier on *real* test data?

Three training sets, one real test set:
  real_only   the real training data (the baseline to beat)
  augmented   real + the synthetic samples the plan calls for (rebalancing or
              per-class augmentation)
  tstr        synthetic data only -- train on synthetic, test on real (Esteban et
              al. 2017); a direct measure of how much class-relevant signal the
              generator captured

Classifier training is stochastic, so each condition is fitted with several
seeds. Gains are reported with a paired bootstrap over test cases (the same
resampled cases for both conditions, averaged over seeds): a 95% interval and
the bootstrap probability that the gain is positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from .features import make_classifier

N_BOOT = 1000


def classifier_description(space) -> str:
    if space.modality == "image":
        return ("CNN: 3 conv blocks (32-64-128 channels, batch norm, ReLU, max-pool) + global "
                "average pooling + linear head; Adam lr 1e-3, weight decay 1e-4, batch 128; early "
                f"stopping on real validation loss (patience 8), cap {space.classifier_seconds:.0f}s")
    return ("multinomial logistic regression (C=1, lbfgs) on standardised evaluation features "
            f"({space.description})")


def concat(a, b):
    if isinstance(a, pd.DataFrame):
        return pd.concat([a, b.astype(a.dtypes.to_dict(), errors="ignore")], ignore_index=True)
    return np.concatenate([a, b])


def take(X, idx):
    return X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[idx]


def fit_predict(space, X, y, X_val, y_val, X_test, n_classes, seeds, weighted=False):
    preds = []
    for s in seeds:
        clf = make_classifier(space, n_classes, s, weighted).fit(X, y, X_val, y_val)
        preds.append(clf.predict(X_test))
    return preds


def _target_recall(y, p, targets):
    if not targets:
        return float("nan")
    r = recall_score(y, p, labels=targets, average=None, zero_division=0)
    return float(np.mean(r))


def summarise(preds, y_test, n_classes, targets) -> dict:
    labels = list(range(n_classes))
    f1 = [f1_score(y_test, p, labels=labels, average="macro", zero_division=0) for p in preds]
    bal = [balanced_accuracy_score(y_test, p) for p in preds]
    rec = np.mean([recall_score(y_test, p, labels=labels, average=None, zero_division=0)
                   for p in preds], axis=0)
    return {"macro_f1": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1)),
            "balanced_accuracy": float(np.mean(bal)),
            "target_class_recall": _target_recall(y_test, preds[0], targets) if len(preds) == 1
            else float(np.mean([_target_recall(y_test, p, targets) for p in preds])),
            "per_class_recall": {int(k): float(v) for k, v in zip(labels, rec)},
            "n_seeds": len(preds)}


def paired_gain(preds_new, preds_base, y_test, metric, rng) -> dict:
    """Bootstrap the gain metric(new) - metric(base).

    Two sources of variability are resampled in each draw (a two-level
    bootstrap): test cases (paired -- the same cases for both conditions) and
    classifier-training seeds (independently per condition). Resampling cases
    alone would understate uncertainty whenever seed-to-seed variation is
    comparable to the effect.
    """
    n, k_new, k_base = len(y_test), len(preds_new), len(preds_base)

    def stat(idx, s_new, s_base):
        return (np.mean([metric(y_test[idx], preds_new[s][idx]) for s in s_new])
                - np.mean([metric(y_test[idx], preds_base[s][idx]) for s in s_base]))

    point = stat(np.arange(n), range(k_new), range(k_base))
    boots = np.array([stat(rng.integers(0, n, n), rng.integers(0, k_new, k_new),
                           rng.integers(0, k_base, k_base)) for _ in range(N_BOOT)])
    return {"gain": float(point), "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
            "prob_positive": float((boots > 0).mean())}


def run_utility(space, real, synth_by_model: dict, n_classes: int, targets: list[int],
                seeds: list[int], want_tstr: bool, rng, class_weighted: bool = False) -> dict:
    """real: dict with train/val/test canonical arrays and labels.
    synth_by_model: name -> {"augment": (X, y) | None, "eval": (X, y) | None}."""
    Xtr, ytr, Xv, yv, Xte, yte = (real[k] for k in ("train", "y_train", "val", "y_val",
                                                      "test", "y_test"))
    out = {"real_only": None, "models": {}, "classifier": classifier_description(space)}
    base = fit_predict(space, Xtr, ytr, Xv, yv, Xte, n_classes, seeds)
    out["real_only"] = summarise(base, yte, n_classes, targets)
    macro = lambda y, p: f1_score(y, p, labels=list(range(n_classes)), average="macro",
                                  zero_division=0)
    trec = lambda y, p: _target_recall(y, p, targets)
    weighted = None
    if class_weighted:
        weighted = fit_predict(space, Xtr, ytr, Xv, yv, Xte, n_classes, seeds, weighted=True)
        out["class_weighted"] = summarise(weighted, yte, n_classes, targets)
    for name, s in synth_by_model.items():
        res = {}
        if s.get("augment") is not None and len(s["augment"][1]):
            Xa, ya = s["augment"]
            aug = fit_predict(space, concat(Xtr, Xa), np.concatenate([ytr, ya]), Xv, yv, Xte,
                              n_classes, seeds)
            res["augmented"] = summarise(aug, yte, n_classes, targets)
            res["macro_f1_gain"] = paired_gain(aug, base, yte, macro, rng)
            if targets:
                res["target_class_recall_gain"] = paired_gain(aug, base, yte, trec, rng)
            if weighted is not None:
                # Does synthetic oversampling beat simply reweighting the loss?
                res["vs_class_weighted_macro_f1"] = paired_gain(aug, weighted, yte, macro, rng)
                if targets:
                    res["vs_class_weighted_target_recall"] = paired_gain(aug, weighted, yte,
                                                                         trec, rng)
            res["n_synthetic_added"] = int(len(ya))
        if want_tstr and s.get("eval") is not None:
            Xs, ys = s["eval"]
            if len(np.unique(ys)) > 1:
                cut = int(0.9 * len(ys))           # synthetic validation split: no real leakage
                perm = rng.permutation(len(ys))
                tr, va = perm[:cut], perm[cut:]
                tstr = fit_predict(space, take(Xs, tr), ys[tr], take(Xs, va), ys[va], Xte,
                                   n_classes, seeds)
                res["tstr"] = summarise(tstr, yte, n_classes, targets)
                res["tstr_vs_real_macro_f1"] = paired_gain(tstr, base, yte, macro, rng)
        out["models"][name] = res
    return out
