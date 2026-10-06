"""Train / validation / test splits, label budgets and subsampling.

The test split is touched only by evaluation. When a file ships its own split
(``LoadedData.splits``) we use it, so results are comparable with published
numbers; otherwise we draw a stratified split from the run seed.

A label budget ("only 5% of labels are available") is applied to train *and*
validation: in that scenario one would not have extra labelled data for model
selection either. Every subsample is stratified so rare classes keep at least
one member whenever possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..schema import DataPlan
from .loaders import LoadedData


@dataclass
class Subset:
    X: object                 # same container type as LoadedData.X
    y: np.ndarray | None
    index: np.ndarray         # row indices into the loaded data

    def __len__(self) -> int:
        return len(self.index)


@dataclass
class Splits:
    train: Subset
    val: Subset
    test: Subset
    notes: list[str]


def _take(X, idx):
    return X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[idx]


def _subset(data: LoadedData, idx: np.ndarray) -> Subset:
    idx = np.sort(idx)
    return Subset(_take(data.X, idx), None if data.y is None else data.y[idx], idx)


def _stratified_fraction(idx: np.ndarray, y: np.ndarray | None, fraction: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Keep ``fraction`` of idx, per class, at least one per class."""
    if fraction >= 1:
        return idx
    if y is None:
        k = max(1, int(round(fraction * len(idx))))
        return rng.choice(idx, k, replace=False)
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        k = max(1, int(round(fraction * len(members))))
        keep.append(rng.choice(members, k, replace=False))
    return np.concatenate(keep)


def make_splits(data: LoadedData, plan: DataPlan, seed: int) -> Splits:
    rng = np.random.default_rng(seed)
    n = data.n_samples
    y = data.y
    notes = []

    if plan.use_predefined_splits and data.splits and "train" in data.splits \
            and "test" in data.splits:
        train_idx = data.splits["train"]
        test_idx = data.splits["test"]
        val_idx = data.splits.get("val")
        if val_idx is None:
            train_idx, val_idx = _split(train_idx, y, plan.val_fraction, seed)
        notes.append("used the file's predefined train/val/test split")
    else:
        all_idx = np.arange(n)
        rest, test_idx = _split(all_idx, y, plan.test_fraction, seed)
        train_idx, val_idx = _split(rest, y, plan.val_fraction / (1 - plan.test_fraction), seed)
        notes.append(f"stratified split {1 - plan.test_fraction - plan.val_fraction:.0%}/"
                     f"{plan.val_fraction:.0%}/{plan.test_fraction:.0%} (seed {seed})")

    if plan.label_budget:
        before = len(train_idx)
        train_idx = _stratified_fraction(train_idx, y, plan.label_budget, rng)
        val_idx = _stratified_fraction(val_idx, y, plan.label_budget, rng)
        notes.append(f"label budget {plan.label_budget:.1%}: train {before} -> {len(train_idx)}")
    if plan.max_train_samples and len(train_idx) > plan.max_train_samples:
        frac = plan.max_train_samples / len(train_idx)
        train_idx = _stratified_fraction(train_idx, y, frac, rng)
        val_idx = _stratified_fraction(val_idx, y, frac, rng)
        notes.append(f"subsampled train to {len(train_idx)} (compute cap)")
    test_cap = 4 * (plan.max_train_samples or 10**9)
    if len(test_idx) > test_cap:
        test_idx = _stratified_fraction(test_idx, y, test_cap / len(test_idx), rng)
        notes.append(f"subsampled test to {len(test_idx)} (compute cap)")

    return Splits(_subset(data, train_idx), _subset(data, val_idx), _subset(data, test_idx), notes)


def _split(idx, y, fraction, seed):
    strat = None
    if y is not None:
        counts = np.bincount(y[idx])
        if counts[counts > 0].min() >= 2:
            strat = y[idx]
    a, b = train_test_split(idx, test_size=fraction, random_state=seed, stratify=strat)
    return np.asarray(a), np.asarray(b)
