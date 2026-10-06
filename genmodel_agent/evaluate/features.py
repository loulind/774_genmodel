"""Evaluation spaces and downstream classifiers.

Fidelity metrics are only as meaningful as the space they are computed in.
Inception features (the usual FID backbone) describe ImageNet photographs, not
28x28 histology or ultrasound, and raw pixel or gene space is dominated by
noise. So each modality gets a domain-appropriate space fitted on the *real
training split only*:

    image   penultimate layer (128-d) of a small CNN trained to classify the real
            training images (or PCA of pixels when there are no labels)
    counts  PCA (50) of log-normalised highly-variable-gene expression
    tabular standardised numerics + one-hot categoricals

Every model's output is first mapped to a *canonical* form shared with the real
data (uint8-quantised pixels; log-normalised HVG expression; a DataFrame), so a
count model and a Gaussian model are compared in the same space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from ..models.train import train_loop
from ..preprocess import CountsHVG, Pixels, Tabular, lognorm


class SmallCNN:
    """Three conv blocks + global average pooling; used as feature extractor and
    as the downstream classifier for utility tests."""

    def __init__(self, C: int, n_classes: int, device: str, seed: int = 0):
        torch.manual_seed(seed)
        self.seed, self.device = seed, torch.device(device)
        self.body = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(self.device)
        self.head = nn.Linear(128, n_classes).to(self.device)

    def fit(self, X, y, X_val, y_val, seconds: float, max_epochs: int = 60, class_weights=None):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        Xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        Yv = torch.as_tensor(y_val, dtype=torch.long, device=self.device)
        params = list(self.body.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        W = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32,
                                                               device=self.device)

        def step(idx):
            loss = F.cross_entropy(self.head(self.body(X[idx])), Y[idx], weight=W)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return {"loss": loss.item()}

        def validate():
            return float(F.cross_entropy(self._logits(Xv), Yv))

        self.history = train_loop(step, len(X), 128, max_epochs, seconds, [self.body, self.head],
                                  validate, patience=8, seed=self.seed, device=self.device)
        return self

    @torch.no_grad()
    def _logits(self, X, batch=2048):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return torch.cat([self.head(self.body(X[i:i + batch])) for i in range(0, len(X), batch)])

    def predict(self, X) -> np.ndarray:
        return self._logits(X).argmax(1).cpu().numpy()

    @torch.no_grad()
    def features(self, X, batch=2048) -> np.ndarray:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return torch.cat([self.body(X[i:i + batch]) for i in range(0, len(X), batch)]).cpu().numpy()


class EvalSpace:
    def __init__(self, modality: str, device: str, seed: int, n_hvg: int | None = None,
                 classifier_seconds: float = 60):
        self.modality, self.device, self.seed = modality, device, seed
        self.n_hvg = n_hvg
        self.classifier_seconds = classifier_seconds
        self.description = ""

    # --- fitting ----------------------------------------------------------------
    def fit(self, train, val) -> "EvalSpace":
        if self.modality == "image":
            self.pixels = Pixels().fit(train)
            Xtr, Xv = self.pixels.transform(train.X), self.pixels.transform(val.X)
            if train.y is not None and len(np.unique(train.y)) > 1:
                n_classes = int(max(train.y.max(), val.y.max()) + 1)
                self.cnn = SmallCNN(Xtr.shape[1], n_classes, self.device, self.seed)
                self.cnn.fit(Xtr, train.y, Xv, val.y, self.classifier_seconds)
                acc = float((self.cnn.predict(Xv) == val.y).mean())
                self.description = (f"128-d penultimate features of a CNN trained on the "
                                    f"{len(Xtr)} real training images (val accuracy {acc:.3f})")
            else:
                self.cnn = None
                self.pca = PCA(50, random_state=self.seed).fit(Xtr.reshape(len(Xtr), -1))
                self.description = "PCA(50) of pixels (no labels for a CNN)"
        elif self.modality == "counts":
            self.hvg = CountsHVG(self.n_hvg or 2000, log=True).fit(train)
            L = self.hvg.transform(train.X)
            self.pca = PCA(min(50, *L.shape), random_state=self.seed).fit(L)
            self.description = (f"PCA({self.pca.n_components_}) of log-normalised expression of "
                                f"{len(self.hvg.genes)} highly variable genes")
        else:
            self.tab = Tabular(keep_missing=False).fit(train)
            self.description = "standardised numerics + one-hot categoricals"
        return self

    # --- canonical forms -----------------------------------------------------------
    def real_canonical(self, X):
        if self.modality == "image":
            return self.pixels.transform(X)
        if self.modality == "counts":
            return self.hvg.transform(X)
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def real_counts(self, X) -> np.ndarray | None:
        return self.hvg.counts(X) if self.modality == "counts" else None

    def model_canonical(self, rep, Z):
        """Map a model's raw output (its representation space) to canonical form."""
        if self.modality == "image":
            return self.pixels.transform(rep.inverse(Z))       # quantise to uint8 and back
        if self.modality == "counts":
            if rep.name == "counts_hvg":
                return lognorm(Z)
            return rep.inverse(Z)                             # lognorm_hvg / lognorm_pca
        return rep.inverse(Z)

    # --- features ------------------------------------------------------------------
    def features(self, canonical) -> np.ndarray:
        if self.modality == "image":
            if self.cnn is not None:
                return self.cnn.features(canonical)
            return self.pca.transform(canonical.reshape(len(canonical), -1))
        if self.modality == "counts":
            return self.pca.transform(canonical)
        return self.tab.transform(canonical)


def make_classifier(space: EvalSpace, n_classes: int, seed: int, weighted: bool = False):
    """Downstream classifier for utility tests: CNN for images, multinomial
    logistic regression on evaluation features otherwise (fast, deterministic,
    and well understood -- utility differences then reflect the data, not tuning)."""
    if space.modality == "image":
        return _CNNClassifier(space, n_classes, seed, weighted)
    return _LogisticClassifier(space, seed, weighted)


def balanced_weights(y, n_classes) -> np.ndarray:
    """Inverse-frequency class weights, n / (K * n_k), as sklearn's 'balanced'."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)


class _CNNClassifier:
    def __init__(self, space, n_classes, seed, weighted=False):
        self.space, self.n_classes, self.seed, self.weighted = space, n_classes, seed, weighted

    def fit(self, X, y, X_val, y_val):
        self.model = SmallCNN(X.shape[1], self.n_classes, self.space.device, self.seed)
        w = balanced_weights(y, self.n_classes) if self.weighted else None
        self.model.fit(X, y, X_val, y_val, self.space.classifier_seconds, class_weights=w)
        return self

    def predict(self, X):
        return self.model.predict(X)


class _LogisticClassifier:
    def __init__(self, space, seed, weighted=False):
        self.space, self.seed, self.weighted = space, seed, weighted

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        F_ = self.space.features(X)
        self.scaler = StandardScaler().fit(F_)
        self.model = LogisticRegression(max_iter=3000, C=1.0, random_state=self.seed,
                                        class_weight="balanced" if self.weighted else None)
        self.model.fit(self.scaler.transform(F_), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(self.space.features(X)))
