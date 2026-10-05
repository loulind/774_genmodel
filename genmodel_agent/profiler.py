"""Step 1 of the agent loop: measure the data.

Everything here is mechanical -- counts, ratios, moments. No judgement is applied
and no LLM is involved; the planner decides what the numbers mean. The profiler
must never raise on unusual input: a dataset that is tiny, constant or riddled
with NaN is exactly what the agent should notice and report, so each optional
measurement degrades to ``None`` rather than throwing.

Modality is inferred from content, not from the file name:
  * rank-4 array                                  -> image
  * non-negative integers, mostly zeros, p >= 100 -> counts (NB/ZINB candidates)
  * anything else                                 -> tabular
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .data.loaders import LoadedData
from .schema import (
    CountSummary, DataProfile, LabelSummary, MissingSummary, ShapeSummary,
)

MAX_ROWS_NEIGHBOURS = 3_000
MAX_ROWS_STATS = 20_000
MAX_FEATURES_SHAPE = 300
COUNT_MIN_ZERO_FRACTION = 0.4
COUNT_MIN_FEATURES = 100
DISCRETE_MAX_LEVELS = 10


def _rows(n: int, limit: int, rng) -> np.ndarray:
    return np.arange(n) if n <= limit else np.sort(rng.choice(n, limit, replace=False))


# ---------------------------------------------------------------------------
# Views of the data
# ---------------------------------------------------------------------------

def numeric_matrix(data: LoadedData, rows: np.ndarray | None = None):
    """Rows of X as a 2-D numeric matrix (sparse kept sparse; tables one-hot).

    Missing numeric entries stay NaN here; callers that need finite values
    impute themselves so the missingness count is never contaminated.
    """
    X = data.X
    if rows is not None:
        X = X.iloc[rows] if isinstance(X, pd.DataFrame) else X[rows]
    if isinstance(X, pd.DataFrame):
        num = X.select_dtypes(include="number").astype(np.float32)
        cat = X.select_dtypes(exclude="number")
        parts = [num.to_numpy()]
        if cat.shape[1]:
            parts.append(pd.get_dummies(cat, dummy_na=False).to_numpy(dtype=np.float32))
        return np.hstack(parts)
    if sp.issparse(X):
        return X
    return np.asarray(X).reshape(X.shape[0], -1)


def _embedding_view(data: LoadedData, modality: str, rng) -> np.ndarray:
    """A dense, finite, comparably-scaled matrix for geometry measurements.

    Counts are library-size normalised and log1p-transformed first; on raw counts
    every distance is dominated by sequencing depth rather than biology.
    """
    rows = _rows(data.n_samples, MAX_ROWS_NEIGHBOURS, rng)
    M = numeric_matrix(data, rows)
    if sp.issparse(M):
        M = M.toarray()
    M = np.asarray(M, dtype=np.float64)
    if modality == "counts":
        lib = M.sum(axis=1, keepdims=True)
        M = np.log1p(M / np.maximum(lib, 1) * 1e4)
    elif modality == "image":
        M = M / max(1.0, float(np.nanmax(M)))
    else:
        med = np.nanmedian(M, axis=0)
        M = np.where(np.isnan(M), np.where(np.isnan(med), 0.0, med), M)
        sd = M.std(axis=0)
        M = (M - M.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return M


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------

NEAR_DUPLICATE_RATIO = 0.2   # NN distance < 20% of the median NN distance


def _two_nn_dimension(M: np.ndarray) -> tuple[float | None, float | None]:
    """Two-NN intrinsic dimension (Facco et al., 2017), linear-fit form.

    mu = r2/r1 is Pareto(d), so -log(1 - F(mu)) = d log(mu): d is the slope of a
    line through the origin. Following the paper we drop the largest 10% of mu,
    which come from near-duplicate points (tiny r1) and otherwise drag the
    closed-form MLE towards absurdly low values (it returned 1.5 on an image set).

    Points whose nearest neighbour is far closer than typical are near-duplicates
    (e.g. repeated frames of one scan); the estimator assumes none exist, so they
    are excluded, and their share is returned as a profile fact in its own right
    -- it bears directly on memorization checks and train/test leakage.

    Returns (dimension, near_duplicate_fraction).
    """
    if M.shape[0] < 10:
        return None, None
    try:
        dist, _ = NearestNeighbors(n_neighbors=3).fit(M).kneighbors(M)
    except Exception:
        return None, None
    r1, r2 = dist[:, 1], dist[:, 2]
    near_dup = r1 < NEAR_DUPLICATE_RATIO * np.median(r1)
    ok = ~near_dup & (r2 > r1)
    if ok.sum() < 10:
        return None, float(near_dup.mean())
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = slice(0, int(0.9 * n))
    x, y = np.log(mu[keep]), -np.log(1 - F[keep])
    return float((x @ y) / (x @ x)), float(near_dup.mean())


def _pca_components_90(M: np.ndarray) -> int | None:
    k = int(min(200, M.shape[0] - 1, M.shape[1]))
    if k < 2:
        return None
    try:
        ratios = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(M) \
            .explained_variance_ratio_
    except Exception:
        return None
    cum = np.cumsum(ratios)
    return int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else k


def _labels(data: LoadedData) -> LabelSummary | None:
    if data.y is None:
        return None
    codes, counts = np.unique(data.y, return_counts=True)
    names = data.class_names or [str(c) for c in codes]
    by_name = {names[c]: int(k) for c, k in zip(codes, counts)}
    i_min = int(np.argmin(counts))
    return LabelSummary(
        n_classes=len(codes), class_names=[names[c] for c in codes], class_counts=by_name,
        imbalance_ratio=float(counts.max() / counts.min()),
        minority_class=names[codes[i_min]], minority_count=int(counts[i_min]),
        minority_fraction=float(counts[i_min] / counts.sum()),
    )


def _missing(data: LoadedData) -> MissingSummary:
    X = data.X
    if isinstance(X, pd.DataFrame):
        mask = X.isna().to_numpy()
    elif sp.issparse(X):
        mask = np.zeros((1, X.shape[1]), dtype=bool)
        if np.issubdtype(X.dtype, np.floating):
            nan_cols = np.zeros(X.shape[1], dtype=bool)
            nan_cols[X.indices[np.isnan(X.data)]] = True
            mask = nan_cols[None, :]
    elif np.issubdtype(np.asarray(X).dtype, np.floating):
        mask = np.isnan(np.asarray(X).reshape(X.shape[0], -1))
    else:
        mask = np.zeros((1, int(np.prod(X.shape[1:]))), dtype=bool)
    per_feature = mask.mean(axis=0)
    return MissingSummary(
        overall_fraction=float(mask.mean()),
        n_features_with_missing=int((per_feature > 0).sum()),
        max_feature_fraction=float(per_feature.max()) if per_feature.size else 0.0,
        rows_with_any_missing_fraction=float(mask.any(axis=1).mean()),
    )


COUNT_MIN_DETECTION = 0.05   # genes detected in >= 5% of cells are assessed
MAX_GENES_LIKELIHOOD = 300
MAX_CELLS_LIKELIHOOD = 3_000


def _nb_vs_zinb(Y: np.ndarray) -> tuple[float, float]:
    """Per-gene maximum-likelihood NB and ZINB fits; returns
    (fraction of genes where ZINB has lower AIC, median fitted zero-inflation pi).

    Moment matching cannot separate zero-inflation from overdispersion -- extra
    zeros simply inflate the variance, the fitted NB gets a smaller theta, and
    its predicted zero fraction rises to meet the observed one. A likelihood fit
    can, because ZINB places the extra mass exactly at zero while NB must spread
    it across the whole distribution. All genes are fitted jointly (their
    parameters are independent, so the summed loss decouples) with L-BFGS.
    """
    import torch

    y = torch.as_tensor(Y, dtype=torch.float64)
    m = y.mean(0).clamp_min(1e-3)

    def nb_logpmf(log_mu, log_theta):
        mu, theta = log_mu.exp(), log_theta.exp()
        return (torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
                + theta * (log_theta - torch.logaddexp(log_theta, log_mu))
                + y * (log_mu - torch.logaddexp(log_theta, log_mu)))

    def fit(zero_inflated: bool):
        log_mu = m.log().clone().requires_grad_(True)
        log_theta = torch.zeros_like(m, requires_grad=True)
        logit_pi = torch.full_like(m, -3.0, requires_grad=True)
        params = [log_mu, log_theta] + ([logit_pi] if zero_inflated else [])
        opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")

        def loglik():
            ll = nb_logpmf(log_mu, log_theta.clamp(-8, 8))
            if zero_inflated:
                log_pi = torch.nn.functional.logsigmoid(logit_pi)
                log_1mpi = torch.nn.functional.logsigmoid(-logit_pi)
                ll = torch.where(y == 0, torch.logaddexp(log_pi, log_1mpi + ll), log_1mpi + ll)
            return ll.sum(0)

        def closure():
            opt.zero_grad()
            loss = -loglik().sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return loglik(), torch.sigmoid(logit_pi)

    ll_nb, _ = fit(False)
    ll_zinb, pi = fit(True)
    aic_gain = 2 * (ll_zinb - ll_nb) - 2          # one extra parameter
    return float((aic_gain > 0).double().mean()), float(pi.median())


def _count_summary(M) -> CountSummary | None:
    """Compare observed zeros to Poisson- and NB-expected zeros, gene by gene.

    NB dispersion is estimated by method of moments, theta = mu^2 / (var - mu).
    If the observed zero fraction clearly exceeds the NB expectation, a
    zero-inflated likelihood is worth its extra parameter; otherwise NB suffices.
    Only genes detected in >= 5% of cells are assessed: near-silent genes are
    almost all zeros under any model and would dilute the comparison.
    """
    try:
        M = M.tocsc() if sp.issparse(M) else np.asarray(M, dtype=np.float64)
        lib = np.asarray(M.sum(axis=1)).ravel()
        mu = np.asarray(M.mean(axis=0)).ravel()
        if sp.issparse(M):
            sq = np.asarray(M.multiply(M).mean(axis=0)).ravel()
            nnz = np.asarray((M != 0).sum(axis=0)).ravel()
        else:
            sq = (M ** 2).mean(axis=0)
            nnz = (M != 0).sum(axis=0)
        var = sq - mu ** 2
        expressed = (mu > 0) & (nnz / M.shape[0] >= COUNT_MIN_DETECTION)
        if expressed.sum() < 10:
            expressed = mu > 0
        mu, var = mu[expressed], var[expressed]
        zero_obs = 1 - nnz[expressed] / M.shape[0]
        theta = np.where(var > mu, mu ** 2 / np.maximum(var - mu, 1e-12), np.inf)
        try:
            rng = np.random.default_rng(0)
            gene_idx = np.flatnonzero(expressed)
            gene_idx = rng.choice(gene_idx, min(MAX_GENES_LIKELIHOOD, len(gene_idx)), replace=False)
            cell_idx = _rows(M.shape[0], MAX_CELLS_LIKELIHOOD, rng)
            Y = M[cell_idx][:, gene_idx]
            Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
            zinb_frac, zinb_pi = _nb_vs_zinb(Y)
        except Exception:
            zinb_frac, zinb_pi = None, None
        p0_pois = np.exp(-mu)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p0_nb = np.where(np.isinf(theta), p0_pois,
                             np.exp(theta * np.log(theta / (theta + mu))))
        return CountSummary(
            mean_library_size=float(lib.mean()), median_library_size=float(np.median(lib)),
            fraction_overdispersed_features=float((var > mu).mean()),
            median_dispersion_ratio=float(np.median(var / mu)),
            observed_zero_fraction=float(zero_obs.mean()),
            poisson_expected_zero_fraction=float(p0_pois.mean()),
            nb_expected_zero_fraction=float(p0_nb.mean()),
            excess_zeros_vs_nb=float(zero_obs.mean() - p0_nb.mean()),
            n_features_assessed=int(expressed.sum()),
            fraction_genes_zinb_preferred=zinb_frac,
            median_zero_inflation_pi=zinb_pi,
        )
    except Exception:
        return None


def _shape(M: np.ndarray, rng) -> ShapeSummary:
    """Skewness, kurtosis and bimodality across (a sample of) continuous columns."""
    try:
        if M.shape[1] == 0:
            return ShapeSummary()
        cols = rng.choice(M.shape[1], min(MAX_FEATURES_SHAPE, M.shape[1]), replace=False)
        sub = M[:, cols]
        keep = np.nanstd(sub, axis=0) > 0
        sub = sub[:, keep]
        if sub.shape[1] == 0:
            return ShapeSummary()
        skew = stats.skew(sub, axis=0, nan_policy="omit")
        kurt = stats.kurtosis(sub, axis=0, nan_policy="omit")
        skew, kurt = np.asarray(skew, float), np.asarray(kurt, float)
        n = np.sum(~np.isnan(sub), axis=0).astype(float)
        bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return ShapeSummary(
            median_abs_skew=float(np.nanmedian(np.abs(skew))),
            fraction_heavily_skewed=float(np.nanmean(np.abs(skew) > 1)),
            fraction_multimodal=float(np.nanmean(bc > 0.555)),
            median_excess_kurtosis=float(np.nanmedian(kurt)),
        )
    except Exception:
        return ShapeSummary()


def _feature_types(data: LoadedData) -> dict:
    out = dict(n_continuous=0, n_discrete=0, n_binary=0, n_categorical=0,
               categorical_cardinalities={})
    X = data.X
    if not isinstance(X, pd.DataFrame):
        return out
    for col in X.columns:
        s = X[col].dropna()
        if not pd.api.types.is_numeric_dtype(X[col]):
            out["n_categorical"] += 1
            out["categorical_cardinalities"][str(col)] = int(s.nunique())
            continue
        levels = s.nunique()
        if levels <= 2:
            out["n_binary"] += 1
        elif levels <= DISCRETE_MAX_LEVELS and np.allclose(s, np.round(s)):
            out["n_discrete"] += 1
        else:
            out["n_continuous"] += 1
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def profile(data: LoadedData, seed: int = 0) -> DataProfile:
    rng = np.random.default_rng(seed)
    n = data.n_samples
    stat_rows = _rows(n, MAX_ROWS_STATS, rng)
    M = numeric_matrix(data, stat_rows)
    p = M.shape[1]

    values = M.data if sp.issparse(M) else np.asarray(M, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)] if values.size else values
    n_cells = M.shape[0] * M.shape[1]
    nonzero = (M.nnz if sp.issparse(M) else int(np.count_nonzero(np.nan_to_num(M))))
    zero_fraction = 1 - nonzero / n_cells if n_cells else 0.0
    is_integer = bool(finite.size and np.all(np.mod(finite, 1) == 0))
    if sp.issparse(M) and M.nnz < n_cells:
        finite = np.append(finite, 0.0)   # implicit zeros are values too
    nonneg = bool(finite.size and finite.min() >= 0)
    has_text = isinstance(data.X, pd.DataFrame) and \
        data.X.select_dtypes(exclude="number").shape[1] > 0

    if data.is_image:
        modality = "image"
        evidence = f"rank-4 array {list(data.X.shape[1:])} (H, W, C)"
    elif (not has_text and is_integer and nonneg and zero_fraction >= COUNT_MIN_ZERO_FRACTION
          and p >= COUNT_MIN_FEATURES):
        modality = "counts"
        evidence = (f"non-negative integers, {100 * zero_fraction:.0f}% zeros, p={p} "
                    f"(>= {COUNT_MIN_FEATURES})")
    else:
        modality = "tabular"
        evidence = ("mixed numeric/categorical columns" if has_text
                    else "numeric matrix not meeting image or count criteria")

    # Variables as the user sees them; one-hot expansion is reported separately.
    n_vars = data.X.shape[1] if isinstance(data.X, pd.DataFrame) else p
    types = _feature_types(data)
    if not isinstance(data.X, pd.DataFrame):
        types["n_continuous"] = p   # pixels / genes are treated as one homogeneous block

    emb = _embedding_view(data, modality, rng)
    if modality == "tabular" and isinstance(data.X, pd.DataFrame):
        cont = data.X.select_dtypes(include="number").to_numpy(dtype=np.float64)[stat_rows]
    elif modality == "counts":
        cont = emb
    else:
        cont = np.asarray(M, dtype=np.float64) if not sp.issparse(M) else emb

    intrinsic_dim, near_dup = _two_nn_dimension(emb)
    try:
        dense_rows = emb[: min(5_000, len(emb))]
        dup = 1 - len(np.unique(np.round(dense_rows, 6), axis=0)) / len(dense_rows)
    except Exception:
        dup = None

    return DataProfile(
        name=data.name, source=data.source, container=data.container,
        modality=modality, modality_evidence=evidence,
        n_samples=n, n_features=n_vars, n_over_p=round(n / max(n_vars, 1), 3),
        n_encoded_features=p,
        image_shape=list(data.X.shape[1:]) if data.is_image else None,
        **types,
        value_min=float(finite.min()) if finite.size else None,
        value_max=float(finite.max()) if finite.size else None,
        is_integer_valued=is_integer, is_nonnegative=nonneg,
        zero_fraction=round(float(zero_fraction), 4),
        is_sparse_storage=sp.issparse(data.X),
        labels=_labels(data), missing=_missing(data),
        counts=_count_summary(M) if modality == "counts" else None,
        shape=_shape(cont, rng),
        intrinsic_dimension=intrinsic_dim,
        near_duplicate_fraction=None if near_dup is None else round(near_dup, 4),
        pca_components_90=_pca_components_90(emb),
        duplicate_fraction=None if dup is None else round(float(dup), 4),
        predefined_splits={k: int(len(v)) for k, v in data.splits.items()} if data.splits else None,
        dense_memory_mb=round(n * p * 4 / 1e6, 1),
    )
