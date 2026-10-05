"""Profiler tests on small synthetic files with known properties."""

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from genmodel_agent.data.loaders import load_path
from genmodel_agent.profiler import profile

rng = np.random.default_rng(0)


def _nb_counts(n, p, zero_inflation=0.0):
    mu = rng.gamma(0.5, 2.0, size=p)
    theta = 2.0
    lam = rng.gamma(theta, mu / theta, size=(n, p))
    X = rng.poisson(lam)
    X[rng.random((n, p)) < zero_inflation] = 0
    return X


def test_image_npz_with_splits(tmp_path):
    f = tmp_path / "imgs.npz"
    np.savez(f, train_images=rng.integers(0, 256, (60, 8, 8), dtype=np.uint8),
             train_labels=np.repeat([0, 1, 2], 20)[:, None],
             test_images=rng.integers(0, 256, (12, 8, 8), dtype=np.uint8),
             test_labels=np.repeat([0, 1, 2], 4)[:, None])
    data = load_path(f)
    pr = profile(data)
    assert pr.modality == "image" and pr.image_shape == [8, 8, 1]
    assert pr.predefined_splits == {"train": 60, "test": 12}
    assert pr.labels.n_classes == 3 and pr.labels.imbalance_ratio == 1.0


def test_mixed_table_with_missing(tmp_path):
    f = tmp_path / "table.csv"
    n = 80
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": rng.normal(50, 10, n).round(1),
        "grade": rng.integers(1, 4, n),
        "smoker": rng.choice(["yes", "no", "?"], n),
        "marker": [f" {v:.2f}" for v in rng.lognormal(size=n)],   # numbers stored as text
        "class": rng.choice(["a", "b"], n, p=[0.8, 0.2]),
    })
    df.loc[:9, "age"] = np.nan
    df.to_csv(f, index=False)
    data = load_path(f)
    pr = profile(data)
    assert pr.modality == "tabular"
    assert "id" not in data.feature_names and pr.n_features == 4
    assert pr.n_categorical == 1 and pr.n_discrete == 1 and pr.n_continuous == 2
    assert pr.missing.n_features_with_missing == 2      # age and smoker ('?')
    assert pr.labels.minority_class == "b"


@pytest.mark.parametrize("zi, expect_excess", [(0.0, False), (0.4, True)])
def test_count_matrix_zero_inflation(tmp_path, zi, expect_excess):
    X = _nb_counts(400, 300, zero_inflation=zi)
    f = tmp_path / "counts.h5ad"
    ad = anndata.AnnData(sp.csr_matrix(X.astype(np.float32)))
    ad.obs["cell_type"] = rng.choice(["t", "b", "rare"], 400, p=[0.6, 0.38, 0.02])
    ad.write_h5ad(f)
    pr = profile(load_path(f))
    assert pr.modality == "counts" and pr.value_min == 0.0
    assert pr.counts.fraction_overdispersed_features > 0.8
    # The likelihood comparison, not the moment-based excess, is the ZI signal.
    assert (pr.counts.median_zero_inflation_pi > 0.1) == expect_excess
