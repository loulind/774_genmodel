"""Memorization: are synthetic samples closer to training points than new data would be?

Distance to closest record (DCR). For each synthetic sample, its distance to the
nearest *training* point; for reference, the same for real *holdout* points.
If the generator produces genuinely new samples, the two distributions should be
similar. If synthetic DCRs are systematically smaller, the model is reproducing
(near-)copies of its training data.

Two summaries:
  * dcr_ratio = median synthetic DCR / median holdout DCR (well below 1: suspicious)
  * too_close_fraction = share of synthetic samples nearer to the training set
    than the 5th percentile of holdout DCRs. Without memorization this is ~0.05;
    we flag it above twice that.
A one-sided Mann-Whitney test of "synthetic DCRs are smaller" gives a p-value.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

EXPECTED_TOO_CLOSE = 0.05
FLAG_MULTIPLE = 2.0


def memorization(train: np.ndarray, holdout: np.ndarray, synth: np.ndarray) -> dict:
    nn = NearestNeighbors(n_neighbors=1).fit(train)
    d_syn = nn.kneighbors(synth)[0][:, 0]
    d_hold = nn.kneighbors(holdout)[0][:, 0]
    threshold = np.quantile(d_hold, EXPECTED_TOO_CLOSE)
    too_close = float((d_syn < threshold).mean())
    p = stats.mannwhitneyu(d_syn, d_hold, alternative="less").pvalue if len(d_syn) > 1 else None
    return {
        "dcr_median_synth": float(np.median(d_syn)),
        "dcr_median_holdout": float(np.median(d_hold)),
        "dcr_ratio": float(np.median(d_syn) / max(np.median(d_hold), 1e-12)),
        "too_close_fraction": too_close,
        "exact_copy_fraction": float((d_syn < 1e-6).mean()),
        "mannwhitney_p_synth_closer": None if p is None else float(p),
        "memorization_flag": bool(too_close > FLAG_MULTIPLE * EXPECTED_TOO_CLOSE),
    }
