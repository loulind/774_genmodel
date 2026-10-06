"""Fidelity and diversity of synthetic samples, measured in a feature space.

All functions take real and synthetic feature matrices (rows = samples) and
return plain floats. Each is also computed for a *real holdout vs real train*
pair, which is the reference: a synthetic set cannot be expected to beat fresh
real data from the same population, so that row is the noise floor against
which the synthetic rows are read.

  precision / recall   Kynkaanniemi et al. (2019): the real (synthetic) manifold
                       is approximated by k-NN balls; precision is the share of
                       synthetic points inside the real manifold (fidelity),
                       recall the share of real points inside the synthetic one
                       (diversity / mode coverage).
  density / coverage   Naeem et al. (2020): outlier-robust versions; coverage is
                       the share of real points whose k-NN ball contains a
                       synthetic sample.
  frechet_distance     Distance between Gaussians fitted to the two feature
                       sets (the FID formula; Heusel et al. 2017) -- here with a
                       domain-appropriate feature extractor, not Inception.
  mmd                  Unbiased squared MMD with an RBF kernel, bandwidth by the
                       median heuristic (Gretton et al. 2012).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, stats
from sklearn.neighbors import NearestNeighbors

K_NN = 5


def _kth_radius(X: np.ndarray, k: int) -> np.ndarray:
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return d[:, -1]


def _ball_counts(points: np.ndarray, centres: np.ndarray, radii: np.ndarray,
                 chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each point: how many centre-balls contain it, and distance to nearest centre.
    Exact (chunked pairwise distances), so no ball is missed however large."""
    from sklearn.metrics import pairwise_distances

    counts = np.zeros(len(points), int)
    nearest = np.full(len(points), np.inf)
    for i in range(0, len(points), chunk):
        D = pairwise_distances(points[i:i + chunk].astype(np.float32),
                               centres.astype(np.float32))
        counts[i:i + chunk] = (D <= radii[None, :]).sum(1)
        nearest[i:i + chunk] = D.min(1)
    return counts, nearest


def precision_recall_density_coverage(real: np.ndarray, synth: np.ndarray,
                                      k: int = K_NN) -> dict:
    r_real = _kth_radius(real, k)
    r_synth = _kth_radius(synth, k)
    in_real, _ = _ball_counts(synth, real, r_real)
    in_synth, _ = _ball_counts(real, synth, r_synth)
    # coverage: real point i is covered if its own k-NN ball holds a synthetic point
    _, nearest_synth = _ball_counts(real, synth, np.zeros(len(synth)))
    return {"precision": float((in_real > 0).mean()),
            "recall": float((in_synth > 0).mean()),
            "density": float(in_real.mean() / k),
            "coverage": float((nearest_synth <= r_real).mean())}


def frechet_distance(real: np.ndarray, synth: np.ndarray) -> float:
    mu1, mu2 = real.mean(0), synth.mean(0)
    s1, s2 = np.cov(real, rowvar=False), np.cov(synth, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def mmd_rbf(real: np.ndarray, synth: np.ndarray, rng: np.random.Generator,
            max_n: int = 2000) -> float:
    if len(real) > max_n:
        real = real[rng.choice(len(real), max_n, replace=False)]
    if len(synth) > max_n:
        synth = synth[rng.choice(len(synth), max_n, replace=False)]
    Z = np.vstack([real, synth])
    sq = (Z ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    bandwidth = np.median(D[np.triu_indices_from(D, 1)])
    Kmat = np.exp(-D / max(bandwidth, 1e-12))
    n, m = len(real), len(synth)
    Kxx, Kyy, Kxy = Kmat[:n, :n], Kmat[n:, n:], Kmat[:n, n:]
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
                 + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


def correlation_difference(real: np.ndarray, synth: np.ndarray, max_features: int = 200) -> float:
    """Mean absolute difference between off-diagonal correlation entries."""
    keep = np.argsort(-real.var(0))[:max_features]
    keep = keep[(real[:, keep].std(0) > 0) & (synth[:, keep].std(0) > 0)]
    if len(keep) < 2:
        return float("nan")
    cr = np.corrcoef(real[:, keep], rowvar=False)
    cs = np.corrcoef(synth[:, keep], rowvar=False)
    iu = np.triu_indices(len(keep), 1)
    return float(np.nanmean(np.abs(cr[iu] - cs[iu])))


def marginal_ks(real: np.ndarray, synth: np.ndarray, max_features: int = 200) -> float:
    """Mean two-sample Kolmogorov-Smirnov statistic over (the most variable) features."""
    keep = np.argsort(-real.var(0))[:max_features]
    return float(np.mean([stats.ks_2samp(real[:, j], synth[:, j]).statistic for j in keep]))


def count_marginals(real_counts: np.ndarray, synth_counts: np.ndarray) -> dict:
    """Gene-wise agreement for count data: means, variances, zero fractions, depth."""
    def logm(x):
        return np.log1p(x)
    mr, ms = real_counts.mean(0), synth_counts.mean(0)
    vr, vs = real_counts.var(0), synth_counts.var(0)
    zr, zs = (real_counts == 0).mean(0), (synth_counts == 0).mean(0)
    return {
        "gene_mean_corr": float(np.corrcoef(logm(mr), logm(ms))[0, 1]),
        "gene_var_corr": float(np.corrcoef(logm(vr), logm(vs))[0, 1]),
        "zero_fraction_real": float(zr.mean()), "zero_fraction_synth": float(zs.mean()),
        "gene_zero_fraction_corr": float(np.corrcoef(zr, zs)[0, 1]),
        "library_size_ks": float(stats.ks_2samp(real_counts.sum(1), synth_counts.sum(1)).statistic),
    }


def table_marginals(real_df, synth_df) -> dict:
    """KS for numeric columns, total-variation distance for categorical ones."""
    import pandas as pd

    ks, tv = [], []
    for c in real_df.columns:
        r, s = real_df[c].dropna(), synth_df[c].dropna()
        if not len(r) or not len(s):
            continue
        if pd.api.types.is_numeric_dtype(r):
            ks.append(stats.ks_2samp(r.astype(float), s.astype(float)).statistic)
        else:
            pr = r.astype(str).value_counts(normalize=True)
            ps = s.astype(str).value_counts(normalize=True)
            tv.append(0.5 * pr.subtract(ps, fill_value=0).abs().sum())
    return {"numeric_ks_mean": float(np.mean(ks)) if ks else None,
            "categorical_tv_mean": float(np.mean(tv)) if tv else None}
