"""Common interface for every generative model in the library.

The orchestrator only ever calls these methods, so models are interchangeable:
the critic can swap one for another without touching the rest of the pipeline.

    fit(X, y, X_val, y_val, extras)   train within spec.train_seconds
    sample(n, y)                      draw n samples (in the representation space)
    score(X, y)                       per-sample log-likelihood or bound, if any
    encode(X, y)                      latent means, if the model has a latent space
    impute(X, mask, n_draws)          fill unobserved entries, if supported
"""

from __future__ import annotations

import numpy as np
import torch

from ..schema import ModelSpec


class GenerativeModel:
    supports_score = False
    supports_encode = False
    supports_impute = False

    def __init__(self, spec: ModelSpec, info: dict, n_classes: int = 0,
                 device: str = "cpu", seed: int = 0):
        self.spec = spec
        self.hp = dict(spec.hyperparameters)
        self.info = info
        self.n_classes = n_classes if spec.conditional else 0
        self.device = torch.device(device)
        self.seed = seed
        self.history: dict = {}
        torch.manual_seed(seed)

    # --- helpers -------------------------------------------------------------
    def onehot(self, y) -> torch.Tensor | None:
        if not self.n_classes or y is None:
            return None
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        return torch.nn.functional.one_hot(y, self.n_classes).float()

    def tensor(self, X) -> torch.Tensor:
        return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

    def n_parameters(self) -> int:
        return sum(p.numel() for m in self.modules() for p in m.parameters())

    def modules(self) -> list[torch.nn.Module]:
        return [v for v in vars(self).values() if isinstance(v, torch.nn.Module)]

    def summary(self) -> dict:
        return {"model": self.spec.name, "role": self.spec.role,
                "conditional": self.spec.conditional, "representation": self.spec.representation,
                "hyperparameters": self.hp, "n_parameters": self.n_parameters(),
                **{k: v for k, v in self.history.items() if k != "curve"}}

    # --- interface -------------------------------------------------------------
    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        raise NotImplementedError

    def sample(self, n: int, y=None) -> np.ndarray:
        raise NotImplementedError

    def score(self, X, y=None, extras=None) -> np.ndarray:
        raise NotImplementedError(f"{self.spec.name} has no likelihood")

    def encode(self, X, y=None) -> np.ndarray:
        raise NotImplementedError(f"{self.spec.name} has no encoder")

    def impute(self, X, mask, n_draws: int = 1) -> np.ndarray:
        raise NotImplementedError(f"{self.spec.name} cannot impute")


def sample_labels(n: int, y_train: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Labels for an unconditional-looking sample that matches the class mix."""
    return rng.choice(y_train, size=n, replace=True)
