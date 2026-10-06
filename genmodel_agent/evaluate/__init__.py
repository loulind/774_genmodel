"""Evaluation entry points used by the orchestrator.

``evaluate_generative`` runs the metrics named in the plan for every trained
model, plus a ``reference_real_holdout`` row that applies the same fidelity
metrics to real held-out data. That row is the yardstick: e.g. a Frechet
distance of 3.1 means little alone, but next to a real-vs-real distance of 0.4
it says the synthetic data are clearly distinguishable from real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import fidelity, privacy
from .utility import run_utility, take


def _cap(n: int, limit: int, rng) -> np.ndarray:
    return np.arange(n) if n <= limit else np.sort(rng.choice(n, limit, replace=False))


def _numeric(space, canonical) -> np.ndarray:
    """Matrix for marginal/correlation checks: features for tables, canonical otherwise."""
    if space.modality == "tabular":
        return space.features(canonical)
    if space.modality == "image":
        return canonical.reshape(len(canonical), -1)
    return np.asarray(canonical)


def fidelity_block(space, F_train, F_ref, F_syn, can_train, can_syn, metrics, rng,
                   with_memorization=True) -> dict:
    out = {}
    if "precision_recall" in metrics or "density_coverage" in metrics:
        out.update(fidelity.precision_recall_density_coverage(F_ref, F_syn))
    if "frechet_distance" in metrics:
        out["frechet_distance"] = fidelity.frechet_distance(F_ref, F_syn)
    if "mmd" in metrics:
        out["mmd"] = fidelity.mmd_rbf(F_ref, F_syn, rng)
    if "marginals" in metrics:
        if space.modality == "tabular":
            out.update(fidelity.table_marginals(can_train, can_syn))
        else:
            out["marginal_ks_mean"] = fidelity.marginal_ks(_numeric(space, can_train),
                                                           _numeric(space, can_syn))
    if "correlation" in metrics:
        out["correlation_mean_abs_diff"] = fidelity.correlation_difference(
            _numeric(space, can_train), _numeric(space, can_syn))
    if with_memorization and "memorization" in metrics:
        out.update(privacy.memorization(F_train, F_ref, F_syn))
    return out


def evaluate_generative(space, real: dict, synth: dict, metrics: list[str], targets: list[int],
                        class_names: list[str] | None, n_classes: int, seeds: list[int],
                        want_tstr: bool, max_samples: int, seed: int) -> dict:
    """real: train/val/test canonical + y_*, and optional train_counts.
    synth: model -> {"eval": (canonical, y), "augment": (canonical, y) | None,
                     "counts": raw counts | None}."""
    rng = np.random.default_rng(seed)
    i_tr = _cap(len(real["y_train"]) if real["y_train"] is not None else len(real["train"]),
                max_samples, rng)
    i_te = _cap(len(real["test"]), max_samples, rng)
    can_tr, can_te = take(real["train"], i_tr), take(real["test"], i_te)
    F_tr, F_te = space.features(can_tr), space.features(can_te)
    y_tr = None if real["y_train"] is None else real["y_train"][i_tr]
    y_te = None if real["y_test"] is None else real["y_test"][i_te]

    results: dict = {"feature_space": space.description, "n_real_train_eval": int(len(i_tr)),
                     "n_real_holdout_eval": int(len(i_te)), "models": {}}
    # Real holdout scored against real train exactly as synthetic data will be.
    results["reference_real_holdout"] = fidelity_block(space, F_tr, F_tr, F_te, can_tr, can_te,
                                                       metrics, rng, with_memorization=False)

    for name, s in synth.items():
        can_s, y_s = s["eval"]
        i_s = _cap(len(can_s), max_samples, rng)
        can_s, y_s = take(can_s, i_s), (None if y_s is None else y_s[i_s])
        F_s = space.features(can_s)
        # Frechet distance, MMD and k-NN precision/recall all depend on sample size,
        # so the synthetic set is scored at the *same n as the real holdout*:
        # otherwise a larger synthetic set can "beat" real data through estimator
        # bias alone (observed: FD 10.3 synthetic vs 12.3 real at n 1,846 vs 528).
        i_eq = _cap(len(F_s), len(F_te), rng)
        r = fidelity_block(space, F_tr, F_tr, F_s[i_eq], can_tr, take(can_s, i_eq), metrics, rng)
        r["n_synth_eval"] = int(len(i_eq))
        # Memorization compares synthetic-to-train with *holdout*-to-train distances.
        if "memorization" in metrics:
            r.update(privacy.memorization(F_tr, F_te, F_s))
        if s.get("counts") is not None and real.get("train_counts") is not None:
            r["count_marginals"] = fidelity.count_marginals(real["train_counts"], s["counts"])
        # Rare classes are where copying is most likely: per-class checks.
        per_class = {}
        if y_s is not None and y_tr is not None:
            for k in targets:
                a, b, c = F_tr[y_tr == k], F_s[y_s == k], F_te[y_te == k]
                if min(len(a), len(b)) < 3 or len(c) < 1:
                    continue
                kk = int(min(fidelity.K_NN, len(a) - 1, len(b) - 1))
                pr = fidelity.precision_recall_density_coverage(a, b, k=kk)
                mem = privacy.memorization(a, c, b)
                label = class_names[k] if class_names else str(k)
                per_class[label] = {**pr, "dcr_ratio": mem["dcr_ratio"],
                                    "too_close_fraction": mem["too_close_fraction"],
                                    "n_real_train": int(len(a)), "n_real_holdout": int(len(c)),
                                    "n_synth": int(len(b)), "k_nn": kk}
        if per_class:
            r["target_classes"] = per_class
        results["models"][name] = r

    if real["y_train"] is not None and ("utility_augmented" in metrics or
                                        "utility_tstr" in metrics):
        results["utility"] = run_utility(
            space, real, {n: {"augment": s.get("augment"), "eval": s["eval"]}
                          for n, s in synth.items()},
            n_classes, targets, seeds, want_tstr and "utility_tstr" in metrics, rng)
    return results
