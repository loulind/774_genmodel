"""Fetch every dataset used in the reports and robustness checks into data/.

Run once:  uv run python scripts/prepare_data.py

This is data *preparation*, deliberately outside the agent: it downloads public
files and writes them in standard formats. The agent never sees dataset names,
only the files this script produces. The one non-trivial step is PBMC 3k, whose
raw counts carry no cell-type labels; we attach the published annotations from
scanpy's processed copy (Wolf et al., 2018 tutorial), matching cells by barcode.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from genmodel_agent.data.download import fetch_with_retries, md5sum  # noqa: E402

DATA = ROOT / "data"
PROJECT1_DATA = Path.home() / "Projects" / "774_dimreduce" / "data"
HF_MIRROR = "https://huggingface.co/datasets/albertvillanova/medmnist-v2/resolve/main/data/"


def medmnist(flag: str) -> Path:
    from medmnist import INFO

    dest = DATA / f"{flag}.npz"
    info = INFO[flag]
    if dest.exists() and md5sum(dest) == info["MD5"]:
        print(f"  {dest.name}: present, MD5 ok")
        return dest
    local = PROJECT1_DATA / dest.name
    if local.exists() and md5sum(local) == info["MD5"]:
        shutil.copy2(local, dest)
        print(f"  {dest.name}: copied from Project 1, MD5 ok")
        return dest
    fetch_with_retries([info["url"], HF_MIRROR + dest.name], dest, info["MD5"])
    return dest


def class_names_sidecar(flag: str) -> None:
    """Write <flag>.classes.json (names in label-code order) from MedMNIST's metadata."""
    import json

    from medmnist import INFO

    labels = INFO[flag]["label"]
    names = [labels[str(i)] for i in range(len(labels))]
    (DATA / f"{flag}.classes.json").write_text(json.dumps(names))
    print(f"  {flag}.classes.json: {names}")


def pbmc3k_labeled() -> Path:
    import anndata
    import scanpy as sc

    dest = DATA / "pbmc3k_labeled.h5ad"
    if dest.exists():
        print(f"  {dest.name}: present")
        return dest
    raw_path = PROJECT1_DATA / "pbmc3k_raw.h5ad"
    raw = anndata.read_h5ad(raw_path) if raw_path.exists() else sc.datasets.pbmc3k()
    processed = sc.datasets.pbmc3k_processed()
    cells = processed.obs_names.intersection(raw.obs_names)
    out = raw[cells].copy()
    out.obs["cell_type"] = processed.obs.loc[cells, "louvain"].astype(str).values
    out.var_names_make_unique()
    out.write_h5ad(dest, compression="gzip")
    print(f"  {dest.name}: {out.n_obs} cells x {out.n_vars} genes, "
          f"{out.obs['cell_type'].nunique()} labelled cell types")
    return dest


def chronic_kidney_disease() -> Path:
    """UCI #336 via its OpenML copy (#42972); UCI itself ships it only as a RAR.

    The source has known typing slips: stray tabs and quotes inside values
    (e.g. a label recorded as 'ckd\\t'). We strip them; nothing else is altered.
    """
    from sklearn.datasets import fetch_openml

    dest = DATA / "chronic_kidney_disease.csv"
    if dest.exists():
        print(f"  {dest.name}: present")
        return dest
    df = fetch_openml(data_id=42972, as_frame=True, parser="auto").frame
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].astype("string").str.replace("\t", "").str.strip(" '\"")
            # An export lost the escape on tab characters, leaving levels such
            # as 'tno' beside 'no' and 'ckdt' beside 'ckd'. Map each back.
            levels = set(values.dropna())
            fixes = {v: v[1:] for v in levels if v.startswith("t") and v[1:] in levels}
            fixes |= {v: v[:-1] for v in levels if v.endswith("t") and v[:-1] in levels}
            df[col] = values.replace(fixes)
    df = df.rename(columns={"classification": "class"})
    df.to_csv(dest, index=False)
    print(f"  {dest.name}: {df.shape[0]} rows x {df.shape[1]} columns, "
          f"class counts {df['class'].value_counts().to_dict()}")
    return dest


def mammography() -> Path:
    """OpenML #310 (ODDS anomaly benchmark): 6 features, ~2.3% positive class."""
    from sklearn.datasets import fetch_openml

    dest = DATA / "mammography.csv"
    if dest.exists():
        print(f"  {dest.name}: present")
        return dest
    frame = fetch_openml(data_id=310, as_frame=True, parser="auto").frame
    frame = frame.rename(columns={frame.columns[-1]: "class"})
    frame.to_csv(dest, index=False)
    print(f"  {dest.name}: {frame.shape[0]} rows, class counts "
          f"{frame['class'].value_counts().to_dict()}")
    return dest


def breast_cancer_wisconsin() -> Path:
    from sklearn.datasets import load_breast_cancer

    dest = DATA / "breast_cancer_wisconsin.csv"
    if dest.exists():
        print(f"  {dest.name}: present")
        return dest
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.data.copy()
    df["diagnosis"] = np.array(bunch.target_names)[bunch.target]
    df.to_csv(dest, index=False)
    print(f"  {dest.name}: {df.shape[0]} rows x {df.shape[1] - 1} features")
    return dest


def main() -> None:
    DATA.mkdir(exist_ok=True)
    print("Preparing datasets in", DATA)
    for flag in ("pathmnist", "breastmnist"):
        medmnist(flag)
        class_names_sidecar(flag)
    pbmc3k_labeled()
    chronic_kidney_disease()
    mammography()
    breast_cancer_wisconsin()


if __name__ == "__main__":
    main()
