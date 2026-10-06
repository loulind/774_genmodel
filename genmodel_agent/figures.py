"""Report figures (static PNG, matplotlib).

Colour encodes identity only and follows the entity, not its rank: real data is
always slot 1 (blue), the primary model slot 2 (orange), the comparison model
slot 3 (aqua) -- the first three slots of the reference categorical palette,
which stay distinguishable under colour-vision deficiency for every pair. Grids
and axes are recessive; every multi-series panel has a legend; no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REAL, SLOTS = "#2a78d6", ["#eb6834", "#1baf7a", "#4a3aa7"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def model_colors(names: list[str]) -> dict[str, str]:
    return {n: SLOTS[i % len(SLOTS)] for i, n in enumerate(names)}


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path.name


def image_grid(real: np.ndarray, y_real: np.ndarray, synth: dict, class_names, path: Path,
               per_class: int = 6) -> str:
    """Rows = classes; columns = real | each model. uint8 NHWC images."""
    classes = sorted(np.unique(y_real))
    blocks = [("real", real, y_real)] + [(n, x, y) for n, (x, y) in synth.items()]
    ncol = per_class * len(blocks)
    fig, axes = plt.subplots(len(classes), ncol, figsize=(ncol * 0.42, len(classes) * 0.46 + 0.4),
                             squeeze=False)
    for r, c in enumerate(classes):
        for b, (name, X, y) in enumerate(blocks):
            members = np.flatnonzero(y == c)[:per_class]
            for j in range(per_class):
                ax = axes[r, b * per_class + j]
                ax.axis("off")
                if j < len(members):
                    img = X[members[j]]
                    ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None,
                              vmin=0, vmax=255)
            if b == 0:
                label = class_names[c] if class_names else str(c)
                axes[r, 0].text(-4, img.shape[0] / 2, label[:18], ha="right", va="center",
                                fontsize=7, color=TEXT_2)
    for b, (name, *_rest) in enumerate(blocks):
        axes[0, b * per_class].set_title(name, loc="left", fontsize=8, color=TEXT)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    return _save(fig, path)


def embedding_overlay(F_real, F_synth: dict, path: Path, title: str,
                      y_real=None, y_synth: dict | None = None, highlight: list[int] | None = None,
                      class_names=None) -> str:
    """2-D PCA of evaluation features: real vs each model; optional rare-class panel."""
    from sklearn.decomposition import PCA

    pca = PCA(2, random_state=0).fit(F_real)
    colors = model_colors(list(F_synth))
    panels = 1 + (1 if highlight else 0)
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.2), squeeze=False)
    ax = axes[0, 0]
    R = pca.transform(F_real)
    ax.scatter(R[:, 0], R[:, 1], s=6, c=REAL, alpha=0.45, label="real (train)", linewidths=0)
    for n, F in F_synth.items():
        S = pca.transform(F)
        ax.scatter(S[:, 0], S[:, 1], s=6, c=colors[n], alpha=0.45, label=n, linewidths=0)
    ax.set(title=title, xlabel="PC 1", ylabel="PC 2")
    ax.legend(markerscale=2, loc="best")
    if highlight:
        ax = axes[0, 1]
        ax.scatter(R[:, 0], R[:, 1], s=4, c=GRID, linewidths=0)
        markers = ["o", "^", "s", "D"]
        for i, k in enumerate(highlight):
            name = class_names[k] if class_names else str(k)
            m = markers[i % len(markers)]
            Rk = R[y_real == k]
            ax.scatter(Rk[:, 0], Rk[:, 1], s=22, c=REAL, marker=m, label=f"{name}: real",
                       edgecolors=SURFACE, linewidths=0.8)
            for n, F in F_synth.items():
                if y_synth and y_synth.get(n) is not None:
                    Sk = pca.transform(F[y_synth[n] == k])
                    ax.scatter(Sk[:, 0], Sk[:, 1], s=22, c=colors[n], marker=m,
                               label=f"{name}: {n}", edgecolors=SURFACE, linewidths=0.8)
        ax.set(title="Target classes (other cells grey)", xlabel="PC 1", ylabel="PC 2")
        ax.legend(fontsize=7, loc="best")
    return _save(fig, path)


def count_gene_stats(real_counts, synth_counts: dict, path: Path) -> str:
    """Per-gene log mean and zero fraction: synthetic vs real (identity line)."""
    colors = model_colors(list(synth_counts))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    mr, zr = np.log1p(real_counts.mean(0)), (real_counts == 0).mean(0)
    for n, S in synth_counts.items():
        axes[0].scatter(mr, np.log1p(S.mean(0)), s=4, c=colors[n], alpha=0.5, label=n,
                        linewidths=0)
        axes[1].scatter(zr, (S == 0).mean(0), s=4, c=colors[n], alpha=0.5, label=n, linewidths=0)
        axes[2].hist(S.sum(1), bins=40, histtype="step", color=colors[n], label=n, density=True)
    axes[2].hist(real_counts.sum(1), bins=40, histtype="step", color=REAL, label="real",
                 density=True)
    for ax, (lo, hi) in zip(axes[:2], [(0, mr.max() * 1.05), (0, 1)]):
        ax.plot([lo, hi], [lo, hi], color=TEXT_2, lw=1, ls="--")
    axes[0].set(title="Gene mean, log(1+mean)", xlabel="real", ylabel="synthetic")
    axes[1].set(title="Gene zero fraction", xlabel="real", ylabel="synthetic")
    axes[2].set(title="Library size (counts per cell)", xlabel="total counts", ylabel="density")
    for ax in axes:
        ax.legend(markerscale=3)
    return _save(fig, path)


def table_marginals(real_df, synth: dict, path: Path, max_cols: int = 8) -> str:
    import pandas as pd

    cols = list(real_df.columns)[:max_cols]
    colors = model_colors(list(synth))
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        r = real_df[c].dropna()
        if pd.api.types.is_numeric_dtype(r):
            bins = np.histogram_bin_edges(r.astype(float), bins=25)
            ax.hist(r.astype(float), bins=bins, histtype="step", color=REAL, density=True,
                    label="real")
            for n, df in synth.items():
                ax.hist(df[c].dropna().astype(float), bins=bins, histtype="step",
                        color=colors[n], density=True, label=n)
        else:
            levels = r.astype(str).value_counts(normalize=True)
            x = np.arange(len(levels))
            width = 0.8 / (1 + len(synth))
            ax.bar(x, levels.values, width, color=REAL, label="real")
            for i, (n, df) in enumerate(synth.items()):
                v = df[c].astype(str).value_counts(normalize=True).reindex(levels.index).fillna(0)
                ax.bar(x + (i + 1) * width, v.values, width, color=colors[n], label=n)
            ax.set_xticks(x + width * len(synth) / 2, [str(s)[:8] for s in levels.index],
                          fontsize=7)
        ax.set_title(str(c), loc="left")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def utility_bars(utility: dict, path: Path, class_names=None, targets=None) -> str:
    """Macro-F1 by training condition (mean +/- sd over seeds) and target-class recall."""
    rows = [("real only", utility["real_only"], REAL)]
    if utility.get("class_weighted"):
        rows.append(("class-weighted", utility["class_weighted"], TEXT_2))
    colors = model_colors(list(utility["models"]))
    for n, r in utility["models"].items():
        if "augmented" in r:
            rows.append((f"real + {n}", r["augmented"], colors[n]))
        if "tstr" in r:
            rows.append((f"{n} only (TSTR)", r["tstr"], colors[n]))
    fig, axes = plt.subplots(1, 2 if targets else 1, figsize=(11 if targets else 6, 3.4),
                             squeeze=False)
    ax = axes[0, 0]
    y = np.arange(len(rows))
    ax.barh(y, [r["macro_f1"] for _, r, _ in rows], xerr=[r["macro_f1_sd"] for _, r, _ in rows],
            color=[c for *_, c in rows], height=0.6, error_kw={"ecolor": TEXT_2, "lw": 1})
    ax.set_yticks(y, [n for n, *_ in rows])
    ax.invert_yaxis()
    ax.set(xlabel="macro-F1 on real test set", title="Downstream classifier", xlim=(0, 1))
    for yi, (_, r, _) in zip(y, rows):
        ax.text(r["macro_f1"] + 0.01, yi, f"{r['macro_f1']:.3f}", va="center", fontsize=8,
                color=TEXT)
    if targets:
        ax = axes[0, 1]
        names = [class_names[k] if class_names else str(k) for k in targets]
        width = 0.8 / len(rows)
        x = np.arange(len(targets))
        for i, (n, r, c) in enumerate(rows):
            ax.bar(x + i * width, [r["per_class_recall"].get(k, r["per_class_recall"].get(str(k), 0))
                                   for k in targets], width, color=c, label=n)
        ax.set_xticks(x + width * (len(rows) - 1) / 2, names)
        ax.set(ylabel="recall", title="Target-class recall", ylim=(0, 1.05))
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def training_curves(histories: dict, path: Path) -> str:
    colors = model_colors(list(histories))
    fig, axes = plt.subplots(1, len(histories), figsize=(4.2 * len(histories), 3.2), squeeze=False)
    for ax, (n, h) in zip(axes[0], histories.items()):
        curve = h.get("curve", [])
        if not curve:
            ax.axis("off")
            continue
        key = next((k for k in ("loss", "wasserstein") if k in curve[0]), None)
        ep = [c["epoch"] for c in curve]
        if key:
            ax.plot(ep, [c.get(key) for c in curve], color=colors[n], label=f"train {key}")
        val = h.get("val_curve") or [c for c in curve if "val_loss" in c]
        if val:
            ax.plot([v["epoch"] for v in val], [v["val_loss"] for v in val], color=REAL,
                    label="validation")
        ax.set(title=f"{n} ({h.get('stopped_because', '')})", xlabel="epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def method_bars(methods: dict, metric: str, path: Path, title: str, lower_better: bool,
                highlight: list[str]) -> str:
    names = [n for n in methods if methods[n].get(metric) is not None]
    vals = [methods[n][metric] for n in names]
    colors = [SLOTS[highlight.index(n)] if n in highlight else TEXT_2 for n in names]
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel=f"{metric} ({'lower' if lower_better else 'higher'} is better)", title=title)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8, color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def pr_curves(y: np.ndarray, scores: dict, path: Path) -> str:
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(5, 3.8))
    palette = {n: (SLOTS[i] if n != "isolation_forest" else TEXT_2) for i, n in enumerate(scores)}
    for n, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=palette[n], label=n)
    ax.axhline(y.mean(), color=GRID, lw=1, ls="--", label=f"prevalence {y.mean():.3f}")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall on held-out data",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=7)
    return _save(fig, path)
