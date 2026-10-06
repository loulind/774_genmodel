"""Read any supported file into one container the rest of the agent understands.

The loader reports structure (array rank, dtype, which columns are text, which
splits the file ships with) and nothing else. Deciding what the data *is* --
image, count matrix or table -- belongs to the profiler, and deciding what to do
with it belongs to the planner. In particular there is no lookup by dataset
name anywhere in this module: a file is understood only through its contents.

Supported formats::

    .npz        MedMNIST-style {split}_images / {split}_labels, or X / y arrays
    .h5ad       AnnData; raw counts taken from layers["counts"] if present
    .csv/.tsv   mixed-type table; text columns kept as categoricals
    .npy        a bare array, unlabelled
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Column names treated as the label when the caller does not name one. Matched
# case-insensitively. These are naming conventions, not dataset identities.
LABEL_CANDIDATES = (
    "label", "labels", "target", "class", "y", "outcome", "diagnosis",
    "cell_type", "celltype", "cell_types",
)
NA_VALUES = ["?", "", "NA", "N/A", "NaN", "nan", "None", "null"]
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class LoadedData:
    """A dataset in the single shape the pipeline understands.

    ``X`` is one of:
      * ndarray of rank 4, (n, H, W, C) -- images (grayscale gets C=1)
      * ndarray or scipy.sparse matrix of rank 2 -- numeric matrix
      * pandas DataFrame -- table, possibly with categorical columns and NaN
    ``splits`` maps "train"/"val"/"test" to row indices when the file ships its
    own split; the agent then respects it instead of drawing a new one.
    """
    name: str
    source: str
    container: str
    X: Any
    y: np.ndarray | None = None
    class_names: list[str] | None = None
    feature_names: list[str] | None = None
    splits: dict[str, np.ndarray] | None = None
    label_column: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def is_image(self) -> bool:
        return isinstance(self.X, np.ndarray) and self.X.ndim == 4


def _encode_labels(values) -> tuple[np.ndarray, list[str]]:
    categories = pd.Series(values).astype("category")
    return categories.cat.codes.to_numpy().astype(np.int64), \
        [str(c) for c in categories.cat.categories]


def _as_image_array(arr: np.ndarray) -> np.ndarray:
    """(n, H, W) -> (n, H, W, 1); (n, C, H, W) with small C -> (n, H, W, C)."""
    if arr.ndim == 3:
        return arr[..., None]
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _load_npz(path: Path) -> LoadedData:
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.keys())

        # Split-structured file: {split}_images or {split}_X, with labels alongside.
        split_x = {s: k for s in SPLIT_NAMES for k in (f"{s}_images", f"{s}_X", f"{s}_x")
                   if k in keys}
        if split_x:
            xs, ys, splits, start = [], [], {}, 0
            for split, x_key in split_x.items():
                arr = npz[x_key]
                xs.append(arr)
                y_key = next((k for k in (f"{split}_labels", f"{split}_y") if k in keys), None)
                ys.append(npz[y_key].reshape(len(arr), -1) if y_key else None)
                splits[split] = np.arange(start, start + len(arr))
                start += len(arr)
            X = np.concatenate(xs)
            y = None
            if all(v is not None for v in ys):
                y_all = np.concatenate(ys)
                # Multi-label targets are not class labels; keep only single-label.
                y = y_all[:, 0] if y_all.shape[1] == 1 else None
        else:
            x_key = next((k for k in ("X", "x", "data", "features", "images") if k in keys), None)
            if x_key is None:
                raise ValueError(f"{path} has no recognised feature array; found {sorted(keys)}")
            X = npz[x_key]
            y_key = next((k for k in ("y", "labels", "target") if k in keys), None)
            y = npz[y_key].squeeze() if y_key else None
            splits = None

    if X.ndim >= 3:
        X = _as_image_array(X)
    class_names = None
    if y is not None:
        y, class_names = _encode_labels(y)
    return LoadedData(name=path.stem, source=str(path), container="npz", X=X, y=y,
                      class_names=class_names, splits=splits)


def _load_h5ad(path: Path, label_column: str | None) -> LoadedData:
    import anndata

    adata = anndata.read_h5ad(path)
    notes = []
    if "counts" in adata.layers:
        X = adata.layers["counts"]
        notes.append("used layers['counts'] as the data matrix")
    else:
        X = adata.X
    if sp.issparse(X):
        X = sp.csr_matrix(X)

    column = label_column or next(
        (c for c in adata.obs.columns if c.lower() in LABEL_CANDIDATES), None)
    y = class_names = None
    if column is not None and column in adata.obs:
        y, class_names = _encode_labels(adata.obs[column].astype(str).to_numpy())
    return LoadedData(name=path.stem, source=str(path), container="h5ad", X=X, y=y,
                      class_names=class_names, feature_names=list(map(str, adata.var_names)),
                      label_column=column, notes=notes)


def _coerce_numeric_text(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Convert text columns that are really numbers (e.g. ' 42', '\\t?') to numeric.

    A column is converted when at least 95% of its non-missing entries parse as
    numbers; the rest become NaN. Below that threshold it is a real categorical.
    """
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            stripped = df[col].astype("string").str.strip()
            stripped = stripped.mask(stripped.isin(NA_VALUES))
            df[col] = stripped
            parsed = pd.to_numeric(stripped, errors="coerce")
            present = stripped.notna().sum()
            if present and parsed.notna().sum() >= 0.95 * present:
                df[col] = parsed
                notes.append(f"column '{col}' parsed as numeric")
    return df


def _load_table(path: Path, label_column: str | None) -> LoadedData:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, na_values=NA_VALUES, skipinitialspace=True)
    notes: list[str] = []
    df = _coerce_numeric_text(df, notes)

    column = label_column or next(
        (c for c in df.columns if str(c).strip().lower() in LABEL_CANDIDATES), None)
    y = class_names = None
    if column is not None and column in df.columns:
        labels = df.pop(column)
        keep = labels.notna().to_numpy()
        if not keep.all():
            notes.append(f"dropped {int((~keep).sum())} rows with a missing label")
            df, labels = df.loc[keep].reset_index(drop=True), labels[keep]
        y, class_names = _encode_labels(labels.astype(str).str.strip().to_numpy())

    # Identifier columns (all-unique text, or named like an id) carry no
    # distributional information and would be memorised by any generator.
    for col in list(df.columns):
        is_text_key = (not pd.api.types.is_numeric_dtype(df[col])
                       and df[col].nunique() == len(df))
        if is_text_key or str(col).strip().lower() in {"id", "index", "unnamed: 0"}:
            df = df.drop(columns=col)
            notes.append(f"dropped identifier column '{col}'")

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")
    return LoadedData(name=path.stem, source=str(path), container=path.suffix.lstrip(".").lower(),
                      X=df, y=y, class_names=class_names,
                      feature_names=[str(c) for c in df.columns], label_column=column, notes=notes)


def load_path(path: str | Path, label_column: str | None = None) -> LoadedData:
    """Load an arbitrary dataset file. See the module docstring for formats."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = _load_npz(path)
    elif suffix == ".h5ad":
        data = _load_h5ad(path, label_column)
    elif suffix in {".csv", ".tsv"}:
        data = _load_table(path, label_column)
    elif suffix == ".npy":
        X = np.load(path)
        data = LoadedData(name=path.stem, source=str(path), container="npy",
                          X=_as_image_array(X) if X.ndim >= 3 else X)
    else:
        raise ValueError(f"unsupported file type '{suffix}'; "
                         "supported: .npz, .h5ad, .csv, .tsv, .npy")

    # Optional sidecar with human-readable class names, in label-code order:
    # <file stem>.classes.json  ->  ["name for code 0", "name for code 1", ...]
    sidecar = path.with_suffix(".classes.json")
    if sidecar.exists() and data.class_names is not None:
        names = json.loads(sidecar.read_text())
        if all(c.isdigit() for c in data.class_names) and len(names) > max(map(int, data.class_names)):
            data.class_names = [str(names[int(c)]) for c in data.class_names]
            data.notes.append(f"class names from {sidecar.name}")

    if data.feature_names is None:
        p = int(np.prod(data.X.shape[1:]))
        data.feature_names = [f"f{i}" for i in range(p)]
    return data
