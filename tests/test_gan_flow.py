"""GANs and flows on small synthetic problems with known answers (CPU)."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import multivariate_normal

from genmodel_agent.data.splits import Subset
from genmodel_agent.models.registry import REGISTRY, build, clip_hyperparameters
from genmodel_agent.preprocess import build_representation
from genmodel_agent.schema import ModelSpec

rng = np.random.default_rng(1)


def spec(name, seconds=8, conditional=False, **hp):
    clean, _ = clip_hyperparameters(name, hp)
    return ModelSpec(name=name, conditional=conditional, hyperparameters=clean,
                     representation=REGISTRY[name].representations[0], train_seconds=seconds)


def bimodal(n):
    centre = rng.choice([-2.0, 2.0], size=n)
    return np.c_[centre + 0.4 * rng.normal(size=n), 0.5 * centre + 0.4 * rng.normal(size=n)]


@pytest.mark.parametrize("name", ["realnvp", "maf"])
def test_flow_is_invertible_and_beats_gaussian(name):
    X, Xt = bimodal(2000).astype(np.float32), bimodal(1000).astype(np.float32)
    m = build(spec(name, hidden_dim=64, n_blocks=4, batch_size=256, max_epochs=400),
              info={"kind": "vector", "shape": (2,)}, device="cpu")
    m.fit(X, X_val=Xt[:300])
    x = torch.tensor(Xt[:50])
    z, _ = m.flow(x)
    assert torch.allclose(m.flow.inverse(z), x, atol=1e-3)
    gauss = multivariate_normal(X.mean(0), np.cov(X.T)).logpdf(Xt).mean()
    assert m.score(Xt).mean() > gauss + 0.3
    far = np.array([[8.0, -8.0]], np.float32)
    assert m.score(far)[0] < np.percentile(m.score(Xt), 1)
    assert m.sample(100).shape == (100, 2)


def test_mode_specific_roundtrip():
    n = 300
    df = pd.DataFrame({"x": np.r_[rng.normal(-5, 0.3, n // 2), rng.normal(5, 0.3, n // 2)],
                       "grade": rng.integers(1, 4, n), "site": rng.choice(["a", "b", "c"], n)})
    rep = build_representation("mode_specific").fit(Subset(df, None, np.arange(n)))
    back = rep.inverse(rep.transform(df))
    assert np.allclose(back["x"], df["x"], atol=0.05)
    assert (back["grade"] == df["grade"]).all() and (back["site"].astype(str) == df["site"]).all()
    assert sum(1 for *_, k in rep.info()["spans"] if k == "mode") == 1


def test_ctgan_keeps_bimodality_and_rare_category():
    n = 1500
    mode = rng.random(n) < 0.5
    df = pd.DataFrame({"x": np.where(mode, rng.normal(-4, 0.5, n), rng.normal(4, 0.5, n)),
                       "cat": np.where(rng.random(n) < 0.06, "rare", np.where(mode, "lo", "hi"))})
    rep = build_representation("mode_specific").fit(Subset(df, None, np.arange(n)))
    m = build(spec("ctgan", seconds=15, batch_size=200, max_epochs=300), info=rep.info(),
              device="cpu")
    m.fit(rep.transform(df))
    s = rep.inverse(m.sample(2000))
    assert ((s["x"] < 0).mean() > 0.3) and ((s["x"] > 0).mean() > 0.3)   # both modes
    assert 0.01 < (s["cat"] == "rare").mean() < 0.2
    assert s["x"].between(-1.5, 1.5).mean() < 0.2                       # not collapsed to the mean


def test_conditional_wgan_gp_recovers_class_means():
    n = 1200
    y = rng.integers(0, 2, n)
    X = (rng.normal(size=(n, 2)) + np.where(y[:, None] == 1, 3.0, -3.0)).astype(np.float32)
    m = build(spec("wgan_gp", seconds=15, conditional=True, latent_dim=8, hidden_dim=64,
                   batch_size=128, n_critic=5, lr=5e-4, max_epochs=3000), info={"kind": "vector", "shape": (2,)},
              n_classes=2, device="cpu")
    m.fit(X, y)
    s0, s1 = m.sample(500, np.zeros(500, int)), m.sample(500, np.ones(500, int))
    assert np.all(np.abs(s0.mean(0) + 3) < 1.0) and np.all(np.abs(s1.mean(0) - 3) < 1.0)


def test_image_wgan_shapes():
    X = rng.uniform(-1, 1, (64, 1, 8, 8)).astype(np.float32)
    m = build(spec("wgan_gp", seconds=2, latent_dim=8, hidden_dim=32, batch_size=32),
              info={"kind": "image", "shape": (1, 8, 8)}, device="cpu")
    m.fit(X)
    assert m.sample(10).shape == (10, 1, 8, 8)
