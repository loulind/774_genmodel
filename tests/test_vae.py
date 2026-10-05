"""VAE family on small synthetic problems with known structure (CPU, a few seconds each)."""

import numpy as np
import pandas as pd

from genmodel_agent.data.splits import Subset
from genmodel_agent.models.registry import REGISTRY, build, clip_hyperparameters
from genmodel_agent.preprocess import build_representation
from genmodel_agent.schema import ModelSpec

rng = np.random.default_rng(0)


def spec(name, seconds=6, conditional=False, **hp):
    clean, _ = clip_hyperparameters(name, hp)
    return ModelSpec(name=name, conditional=conditional, hyperparameters=clean,
                     representation=REGISTRY[name].representations[0], train_seconds=seconds)


def test_conditional_image_vae_separates_classes():
    n = 400
    y = np.repeat([0, 1], n // 2)
    imgs = np.clip(rng.normal(60 + 140 * y[:, None, None, None], 20, (n, 8, 8, 1)), 0, 255)
    rep = build_representation("pixels").fit(Subset(imgs.astype(np.uint8), y, np.arange(n)))
    X = rep.transform(imgs.astype(np.uint8))
    m = build(spec("vae", conditional=True, latent_dim=4, hidden_dim=64, batch_size=64,
                   max_epochs=60), info=rep.info(), n_classes=2, device="cpu")
    hist = m.fit(X, y)
    assert hist["epochs_run"] >= 1 and m.summary()["n_parameters"] > 0
    s = rep.inverse(m.sample(200, np.repeat([0, 1], 100)))
    assert s.shape == (200, 8, 8, 1) and s.dtype == np.uint8
    assert s[100:].mean() - s[:100].mean() > 60          # class 1 is brighter


def test_nb_vae_reproduces_zero_fraction():
    n, G = 600, 60
    mu = rng.gamma(1.0, 2.0, G)
    lib = rng.lognormal(0, 0.3, n)
    lam = rng.gamma(2.0, (lib[:, None] * mu) / 2.0)
    X = rng.poisson(lam).astype(np.float32)
    s = spec("nb_vae", seconds=10, latent_dim=4, hidden_dim=64, batch_size=128, max_epochs=150)
    m = build(s, info={"kind": "vector", "shape": (G,)}, device="cpu")
    m.fit(X, extras={"library": X.sum(1)}, X_val=X[:100], extras_val={"library": X[:100].sum(1)})
    S = m.sample(n)
    assert S.shape == (n, G) and np.all(S >= 0) and np.allclose(S, np.round(S))
    assert abs((S == 0).mean() - (X == 0).mean()) < 0.05
    assert 0.5 < m.summary()["median_inverse_dispersion_theta"] < 10   # truth: 2


def test_miwae_beats_mean_imputation():
    n, d = 600, 8
    Z = rng.normal(size=(n, 2))
    full = Z @ rng.normal(size=(2, d)) + 0.1 * rng.normal(size=(n, d))
    df = pd.DataFrame(full, columns=[f"v{i}" for i in range(d)])
    df["grp"] = np.where(Z[:, 0] > 0, "hi", "lo")             # categorical tied to latent
    holes = rng.random((n, d)) < 0.2
    obs = df.copy()
    obs.iloc[:, :d] = obs.iloc[:, :d].mask(holes)
    rep = build_representation("standardized_onehot").fit(Subset(obs, None, np.arange(n)))
    X, M = rep.transform(obs), rep.observed_mask(obs)
    m = build(spec("masked_vae", seconds=12, latent_dim=2, hidden_dim=64, batch_size=64,
                   max_epochs=300), info=rep.info(), device="cpu")
    m.fit(X, extras={"mask": M})
    imputed = rep.inverse(m.impute(X, M))
    truth = df.iloc[:, :d].to_numpy()[holes]
    rmse_vae = np.sqrt(np.mean((imputed.iloc[:, :d].to_numpy()[holes] - truth) ** 2))
    rmse_mean = np.sqrt(np.mean((obs.iloc[:, :d].mean().to_numpy()[np.where(holes)[1]] - truth) ** 2))
    assert rmse_vae < 0.6 * rmse_mean
    draws = m.impute(X, M, n_draws=5)
    assert draws.shape == (5, *X.shape) and draws[:, holes[:, :1].ravel(), 0].std(0).mean() > 0
