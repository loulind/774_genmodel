"""Diffusion models on tiny synthetic problems (CPU)."""

import numpy as np
import torch

from genmodel_agent.models.diffusion import cosine_alpha_bar
from genmodel_agent.models.registry import REGISTRY, build, clip_hyperparameters
from genmodel_agent.schema import ModelSpec

rng = np.random.default_rng(2)


def spec(name, seconds, conditional=False, **hp):
    clean, _ = clip_hyperparameters(name, hp)
    return ModelSpec(name=name, conditional=conditional, hyperparameters=clean,
                     representation=REGISTRY[name].representations[0], train_seconds=seconds)


def test_cosine_schedule_monotone():
    a = cosine_alpha_bar(1000)
    assert a[0] > 0.99 and a[-1] < 1e-3 and torch.all(a[1:] <= a[:-1])


def test_vector_ddpm_conditional_bimodal():
    n = 2000
    y = rng.integers(0, 2, n)
    X = (rng.normal(0, 0.3, (n, 2)) + np.where(y[:, None] == 1, 1.5, -1.5)).astype(np.float32)
    m = build(spec("ddpm_mlp", 20, conditional=True, hidden_dim=128, n_layers=3, timesteps=200,
                   sample_steps=50, guidance_scale=1.0, batch_size=256, lr=2e-3,
                   max_epochs=2000), info={"kind": "vector", "shape": (2,)}, n_classes=2,
              device="cpu")
    m.fit(X, y, X_val=X[:200], y_val=y[:200])
    assert m.history["val_curve"], "validation loss should be recorded"
    s0, s1 = m.sample(400, np.zeros(400, int)), m.sample(400, np.ones(400, int))
    assert np.all(np.abs(s0.mean(0) + 1.5) < 0.4) and np.all(np.abs(s1.mean(0) - 1.5) < 0.4)
    assert 0.1 < s1.std(0).mean() < 0.8                     # truth 0.3: neither collapsed nor noise


def test_image_ddpm_runs_and_respects_range():
    X = np.clip(rng.normal(0, 0.5, (64, 1, 8, 8)), -1, 1).astype(np.float32)
    y = rng.integers(0, 3, 64)
    m = build(spec("ddpm_unet", 4, conditional=True, base_channels=16, timesteps=100,
                   sample_steps=10, batch_size=32, max_epochs=5),
              info={"kind": "image", "shape": (1, 8, 8)}, n_classes=3, device="cpu")
    m.fit(X, y)
    s = m.sample(12, np.arange(12) % 3)
    assert s.shape == (12, 1, 8, 8) and s.min() >= -1 and s.max() <= 1
