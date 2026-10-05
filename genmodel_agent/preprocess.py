"""Representations: the data space each model is trained in, with inverses.

Every representation is fitted on the training split only (no test leakage),
maps raw data to a float32 array the models consume, and maps model output back
to the raw container so samples can be saved, plotted and evaluated like real
data. Names match ``ModelInfo.representations`` in the registry.

    pixels               images -> (n, C, H, W) in [-1, 1]
    standardized         table -> z-scored numerics + one-hot categoricals,
                         missing entries filled (median / mode) for models that
                         need complete rows
    standardized_onehot  as above, but missing entries left as 0 with an observed
                         mask, for models that train on observed entries only
    counts_hvg           raw counts of the top highly variable genes (+ library size)
    lognorm_hvg          log1p(counts / library * 1e4) of the same genes
    lognorm_pca          PCA scores of lognorm_hvg (for vector diffusion)
    mode_specific        CTGAN encoding (implemented with the GAN, models/gan.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data.splits import Subset


class Representation:
    name = "base"

    def fit(self, train: Subset) -> "Representation":
        return self

    def transform(self, X) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, Z: np.ndarray):
        raise NotImplementedError

    def info(self) -> dict:
        return {}

    def observed_mask(self, X) -> np.ndarray | None:
        return None

    def extras(self, X) -> dict:
        return {}


class Pixels(Representation):
    name = "pixels"

    def fit(self, train):
        self.shape = train.X.shape[1:]           # H, W, C
        self.max_value = 255.0 if train.X.dtype == np.uint8 or train.X.max() > 1.5 else 1.0
        return self

    def transform(self, X):
        Z = np.asarray(X, dtype=np.float32) / self.max_value * 2 - 1
        return np.ascontiguousarray(Z.transpose(0, 3, 1, 2))

    def inverse(self, Z):
        X = (np.clip(Z, -1, 1) + 1) / 2 * self.max_value
        X = X.transpose(0, 2, 3, 1)
        return np.round(X).astype(np.uint8) if self.max_value == 255.0 else X.astype(np.float32)

    def info(self):
        H, W, C = self.shape
        return {"kind": "image", "shape": (C, H, W)}


class Tabular(Representation):
    """Standardised numerics + one-hot categoricals, with spans for decoders.

    ``spans`` lists (start, end, kind) per original column: kind is "cont" for a
    single standardised column and "cat" for a one-hot block, so decoders can put
    a Gaussian on the former and a softmax on the latter.
    """

    def __init__(self, keep_missing: bool):
        self.keep_missing = keep_missing
        self.name = "standardized_onehot" if keep_missing else "standardized"

    def fit(self, train):
        df = _as_frame(train.X)
        self.columns = list(df.columns)
        self.num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        num = df[self.num_cols].astype(float)
        self.median = num.median()
        self.mean = num.mean()
        self.std = num.std().replace(0, 1).fillna(1)
        self.lo, self.hi = num.min(), num.max()
        self.integer = {c: bool(np.allclose(num[c].dropna(), np.round(num[c].dropna())))
                        for c in self.num_cols}
        self.levels = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.cat_cols}
        self.mode = {c: df[c].dropna().astype(str).mode().iloc[0] if df[c].notna().any() else ""
                     for c in self.cat_cols}
        self.spans, start = [], 0
        for c in self.num_cols:
            self.spans.append((start, start + 1, "cont"))
            start += 1
        for c in self.cat_cols:
            k = max(1, len(self.levels[c]))
            self.spans.append((start, start + k, "cat"))
            start += k
        self.dim = start
        return self

    def _encode(self, X):
        df = _as_frame(X, self.columns)
        num = df[self.num_cols].astype(float)
        num_mask = num.notna().to_numpy()
        if not self.keep_missing:
            num = num.fillna(self.median)
        Z_num = ((num - self.mean) / self.std).fillna(0).to_numpy(np.float32)
        blocks, masks = [Z_num], [num_mask]
        for c in self.cat_cols:
            values = df[c].astype("string")
            observed = values.notna().to_numpy()
            if not self.keep_missing:
                values = values.fillna(self.mode[c])
            onehot = np.zeros((len(df), max(1, len(self.levels[c]))), np.float32)
            lookup = {v: i for i, v in enumerate(self.levels[c])}
            for row, v in enumerate(values):
                if v is not pd.NA and v in lookup:
                    onehot[row, lookup[v]] = 1.0
            blocks.append(onehot)
            masks.append(np.repeat(observed[:, None], onehot.shape[1], axis=1))
        return np.hstack(blocks).astype(np.float32), np.hstack(masks)

    def transform(self, X):
        return self._encode(X)[0]

    def observed_mask(self, X):
        mask = self._encode(X)[1]
        return mask if self.keep_missing else np.ones_like(mask, dtype=bool)

    def inverse(self, Z):
        out = {}
        for c, (a, b, kind) in zip(self.num_cols + self.cat_cols, self.spans):
            if kind == "cont":
                v = Z[:, a] * self.std[c] + self.mean[c]
                v = np.clip(v, self.lo[c], self.hi[c])
                out[c] = np.round(v) if self.integer[c] else v
            else:
                idx = Z[:, a:b].argmax(axis=1)
                out[c] = pd.Categorical(np.array(self.levels[c], dtype=object)[idx],
                                        categories=self.levels[c])
        return pd.DataFrame(out)[self.columns]

    def info(self):
        return {"kind": "vector", "shape": (self.dim,), "spans": self.spans}


class CountsHVG(Representation):
    """Top-k highly variable genes (Seurat dispersion method on log-normalised train)."""
    name = "counts_hvg"

    def __init__(self, n_hvg: int = 2000, log: bool = False):
        self.n_hvg = n_hvg
        self.log = log
        self.name = "lognorm_hvg" if log else "counts_hvg"

    def fit(self, train):
        import anndata
        import scanpy as sc

        X = sp.csr_matrix(train.X, dtype=np.float32)
        ad = anndata.AnnData(X)
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        n_top = min(self.n_hvg, X.shape[1])
        sc.pp.highly_variable_genes(ad, flavor="seurat", n_top_genes=n_top)
        self.genes = np.flatnonzero(ad.var["highly_variable"].to_numpy())
        self.n_genes_total = X.shape[1]
        return self

    def counts(self, X) -> np.ndarray:
        X = X[:, self.genes]
        return (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)

    def transform(self, X):
        C = self.counts(X)
        return lognorm(C) if self.log else C

    def inverse(self, Z):
        """Counts models return counts already; the log model returns lognorm values."""
        return Z

    def extras(self, X):
        return {"library": self.counts(X).sum(axis=1)}

    def info(self):
        return {"kind": "vector", "shape": (len(self.genes),), "genes": self.genes}


class LogNormPCA(Representation):
    name = "lognorm_pca"

    def __init__(self, n_hvg: int = 2000, n_components: int = 50):
        self.hvg = CountsHVG(n_hvg, log=True)
        self.k = n_components

    def fit(self, train):
        from sklearn.decomposition import PCA

        self.hvg.fit(train)
        L = self.hvg.transform(train.X)
        self.pca = PCA(n_components=min(self.k, *L.shape), random_state=0).fit(L)
        self.scale = self.pca.transform(L).std(axis=0).clip(1e-6)
        return self

    def transform(self, X):
        return (self.pca.transform(self.hvg.transform(X)) / self.scale).astype(np.float32)

    def inverse(self, Z):
        return np.clip(self.pca.inverse_transform(Z * self.scale), 0, None).astype(np.float32)

    def info(self):
        return {"kind": "vector", "shape": (self.pca.n_components_,)}


def lognorm(C: np.ndarray, target: float = 1e4) -> np.ndarray:
    lib = C.sum(axis=1, keepdims=True)
    return np.log1p(C / np.maximum(lib, 1) * target).astype(np.float32)


def _as_frame(X, columns=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X if columns is None else X[columns]
    arr = X.toarray() if sp.issparse(X) else np.asarray(X)
    cols = columns or [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def build_representation(name: str, n_hvg: int | None = None, **kw) -> Representation:
    if name == "pixels":
        return Pixels()
    if name == "standardized":
        return Tabular(keep_missing=False)
    if name == "standardized_onehot":
        return Tabular(keep_missing=True)
    if name == "counts_hvg":
        return CountsHVG(n_hvg or 2000, log=False)
    if name == "lognorm_hvg":
        return CountsHVG(n_hvg or 2000, log=True)
    if name == "lognorm_pca":
        return LogNormPCA(n_hvg or 2000)
    if name == "mode_specific":
        from .models.gan import ModeSpecificTabular
        return ModeSpecificTabular(**kw)
    raise ValueError(f"unknown representation '{name}'")
