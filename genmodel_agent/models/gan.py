"""Adversarial models: WGAN-GP (images or vectors) and a CTGAN-style tabular GAN.

Both use the Wasserstein objective with gradient penalty (Gulrajani et al. 2017):
the critic estimates the Wasserstein-1 distance between real and generated
distributions, and the penalty keeps its gradient norm near 1 (the 1-Lipschitz
constraint) without weight clipping. Adam with betas (0.5, 0.9) follows that paper.

CTGAN (Xu et al. 2019) adds three ideas for mixed-type tables, all implemented here:
  * mode-specific normalisation -- each continuous column is modelled by a
    variational Gaussian mixture, and a value is encoded as (which mode, where in
    that mode), so multimodal and skewed columns do not collapse to the mean;
  * a conditional vector -- the generator is told "produce a row whose column j
    equals category k", and a cross-entropy term makes it comply;
  * training-by-sampling -- that (j, k) is drawn with probability proportional to
    log frequency, so rare categories are seen often enough to be learned.

GANs have no likelihood, so there is no validation loss for early stopping;
training stops on the time or epoch cap, and the critic's Wasserstein estimate
is recorded as the training curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..data.splits import Subset
from ..preprocess import Representation, _as_frame
from .base import GenerativeModel
from .train import train_loop
from .vae import mlp, split_spans


def gradient_penalty(critic, real, fake, cond):
    eps = torch.rand(real.shape[0], *([1] * (real.dim() - 1)), device=real.device)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(mix, cond)
    grad = torch.autograd.grad(out.sum(), mix, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


# ---------------------------------------------------------------------------
# WGAN-GP
# ---------------------------------------------------------------------------

class ConvGenerator(nn.Module):
    def __init__(self, z, c, C, H, W, ch):
        super().__init__()
        self.shape = (2 * ch, H // 4, W // 4)
        self.fc = nn.Sequential(nn.Linear(z + c, int(np.prod(self.shape))),
                                nn.BatchNorm1d(int(np.prod(self.shape))), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, C, 4, 2, 1), nn.Tanh())

    def forward(self, z, c=None):
        h = z if c is None else torch.cat([z, c], 1)
        return self.net(self.fc(h).view(-1, *self.shape))


class ConvCritic(nn.Module):
    """No batch norm: the gradient penalty is defined per sample."""

    def __init__(self, c, C, H, W, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.LeakyReLU(0.2), nn.Flatten())
        self.head = nn.Linear(2 * ch * (H // 4) * (W // 4) + c, 1)

    def forward(self, x, c=None):
        h = self.net(x)
        return self.head(h if c is None else torch.cat([h, c], 1)).squeeze(1)


class VectorCritic(nn.Module):
    """MLP critic. CTGAN uses dropout 0.5 in its critic; plain WGAN-GP uses none,
    since dropout adds noise to the per-sample gradient penalty."""

    def __init__(self, d, c, hidden, pac=1, dropout=0.0):
        super().__init__()
        self.pac = pac
        drop = [nn.Dropout(dropout)] if dropout else []
        self.net = nn.Sequential(
            nn.Linear((d + c) * pac, hidden), nn.LeakyReLU(0.2), *drop,
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), *drop,
            nn.Linear(hidden, 1))

    def forward(self, x, c=None):
        h = x if c is None else torch.cat([x, c], 1)
        return self.net(h.reshape(-1, h.shape[1] * self.pac)).squeeze(1)


class WGANGP(GenerativeModel):
    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, n_classes, device, seed)
        hp, c = self.hp, self.n_classes
        self.z = int(hp["latent_dim"])
        self.is_image = info["kind"] == "image" and info["shape"][1] % 4 == 0
        if self.is_image:
            C, H, W = info["shape"]
            ch = max(16, int(hp["hidden_dim"]) // 4)
            self.G = ConvGenerator(self.z, c, C, H, W, ch).to(self.device)
            self.D = ConvCritic(c, C, H, W, ch).to(self.device)
            self.cont_idx, self.cats = None, []
        else:
            d, h = int(info["shape"][0]), int(hp["hidden_dim"])
            self.G = mlp(self.z + c, h, 2, d, norm=True).to(self.device)
            self.D = VectorCritic(d, c, h).to(self.device)
            cont, self.cats = split_spans(info.get("spans"), d)
            self.cont_idx = cont

    def _generate(self, z, c, hard=False):
        if self.is_image:
            return self.G(z, c)
        out = self.G(z if c is None else torch.cat([z, c], 1))
        if self.cats:
            parts = out.clone()
            for a, b in self.cats:
                logits = out[:, a:b]
                parts[:, a:b] = (F.one_hot(logits.argmax(1), b - a).float() if hard
                                 else F.softmax(logits, 1))
            out = parts
        return out

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        X, C = self.tensor(X), self.onehot(y)
        lr = float(self.hp["lr"])
        optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
        optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))
        n_critic, lam = int(self.hp["n_critic"]), float(self.hp["gp_lambda"])
        counter = {"i": 0}

        def step(idx):
            real = X[idx]
            c = None if C is None else C[idx]
            z = torch.randn(len(idx), self.z, device=self.device)
            fake = self._generate(z, c).detach()
            w = self.D(real, c).mean() - self.D(fake, c).mean()
            lossD = -w + lam * gradient_penalty(self.D, real, fake, c)
            optD.zero_grad()
            lossD.backward()
            optD.step()
            counter["i"] += 1
            logs = {"wasserstein": w.item(), "critic_loss": lossD.item()}
            if counter["i"] % n_critic == 0:
                z = torch.randn(len(idx), self.z, device=self.device)
                lossG = -self.D(self._generate(z, c), c).mean()
                optG.zero_grad()
                lossG.backward()
                optG.step()
                logs["generator_loss"] = lossG.item()
            return logs

        self.history = train_loop(step, len(X), int(self.hp["batch_size"]),
                                  int(self.hp["max_epochs"]), self.spec.train_seconds,
                                  [self.G, self.D], None, seed=self.seed, device=self.device)
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None, batch=2048):
        self.G.eval()
        out = []
        for i in range(0, n, batch):
            k = min(batch, n - i)
            z = torch.randn(k, self.z, device=self.device)
            c = self.onehot(None if y is None else np.asarray(y)[i:i + k])
            out.append(self._generate(z, c, hard=True).cpu())
        return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# CTGAN
# ---------------------------------------------------------------------------

class ModeSpecificTabular(Representation):
    """CTGAN's encoding. Continuous column -> [alpha, one-hot(mode)];
    categorical (or integer column with <= 10 levels) -> one-hot.

    Missing values are filled (median / mode) before encoding: like any GAN,
    CTGAN needs complete rows.
    """
    name = "mode_specific"
    DISCRETE_MAX_LEVELS = 10

    def __init__(self, max_modes: int = 5, seed: int = 0):
        self.max_modes, self.seed = int(max_modes), seed

    def fit(self, train: Subset):
        from sklearn.mixture import BayesianGaussianMixture

        df = _as_frame(train.X)
        self.columns = list(df.columns)
        self.meta, self.spans, self.discrete_blocks, start = [], [], [], 0
        for col in self.columns:
            s = df[col]
            numeric = pd.api.types.is_numeric_dtype(s)
            few = numeric and s.dropna().nunique() <= self.DISCRETE_MAX_LEVELS and \
                np.allclose(s.dropna(), np.round(s.dropna()))
            if numeric and not few:
                v = s.fillna(s.median()).to_numpy(float)[:, None]
                bgm = BayesianGaussianMixture(
                    n_components=self.max_modes, weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=1e-3, max_iter=200, random_state=self.seed).fit(v)
                active = np.flatnonzero(bgm.weights_ > 0.005)
                k = len(active)
                self.meta.append({"col": col, "kind": "cont", "bgm": bgm, "active": active,
                                  "median": float(s.median()), "lo": float(s.min()),
                                  "hi": float(s.max())})
                self.spans += [(start, start + 1, "alpha"), (start + 1, start + 1 + k, "mode")]
                start += 1 + k
            else:
                values = s.astype("string")
                levels = sorted(values.dropna().unique(), key=lambda x: (len(x), x)) if not few \
                    else [str(int(v)) for v in sorted(s.dropna().unique())]
                fill = (str(int(s.dropna().mode().iloc[0])) if few else
                        values.dropna().mode().iloc[0]) if s.notna().any() else levels[0]
                self.meta.append({"col": col, "kind": "cat", "levels": levels, "fill": fill,
                                  "numeric": numeric})
                self.spans.append((start, start + len(levels), "cat"))
                self.discrete_blocks.append((start, start + len(levels)))
                start += len(levels)
        self.dim = start
        return self

    def transform(self, X):
        df = _as_frame(X, self.columns)
        rng = np.random.default_rng(self.seed)
        out = np.zeros((len(df), self.dim), np.float32)
        for m, span in zip(self.meta, self._column_spans()):
            if m["kind"] == "cont":
                v = df[m["col"]].astype(float).fillna(m["median"]).to_numpy()
                bgm, act = m["bgm"], m["active"]
                means = bgm.means_.ravel()[act]
                stds = np.sqrt(bgm.covariances_.ravel()[act])
                prob = bgm.predict_proba(v[:, None])[:, act] + 1e-6
                prob /= prob.sum(1, keepdims=True)
                mode = np.array([rng.choice(len(act), p=p) for p in prob])
                alpha = np.clip((v - means[mode]) / (4 * stds[mode]), -0.99, 0.99)
                a = span[0]
                out[:, a] = alpha
                out[np.arange(len(df)), a + 1 + mode] = 1
            else:
                lookup = {lv: i for i, lv in enumerate(m["levels"])}
                vals = df[m["col"]]
                vals = vals.map(lambda x: str(int(x)) if m["numeric"] and pd.notna(x) else x) \
                    if m["numeric"] else vals.astype("string")
                for row, v in enumerate(vals):
                    v = m["fill"] if (v is None or v is pd.NA or (isinstance(v, float)
                                                                   and np.isnan(v))) else v
                    out[row, span[0] + lookup.get(str(v), 0)] = 1
        return out

    def _column_spans(self):
        spans, i = [], 0
        for m in self.meta:
            if m["kind"] == "cont":
                spans.append((self.spans[i][0], self.spans[i + 1][1]))
                i += 2
            else:
                spans.append(self.spans[i][:2])
                i += 1
        return spans

    def inverse(self, Z):
        out = {}
        for m, (a, b) in zip(self.meta, self._column_spans()):
            if m["kind"] == "cont":
                bgm, act = m["bgm"], m["active"]
                mode = Z[:, a + 1:b].argmax(1)
                means = bgm.means_.ravel()[act]
                stds = np.sqrt(bgm.covariances_.ravel()[act])
                v = np.clip(Z[:, a], -1, 1) * 4 * stds[mode] + means[mode]
                out[m["col"]] = np.clip(v, m["lo"], m["hi"])
            else:
                labels = np.array(m["levels"], dtype=object)[Z[:, a:b].argmax(1)]
                out[m["col"]] = labels.astype(float) if m["numeric"] else \
                    pd.Categorical(labels, categories=m["levels"])
        return pd.DataFrame(out)[self.columns]

    def info(self):
        return {"kind": "vector", "shape": (self.dim,), "spans": self.spans,
                "discrete_blocks": self.discrete_blocks}


class Residual(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc, self.bn = nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out)

    def forward(self, x):
        return torch.cat([F.relu(self.bn(self.fc(x))), x], 1)


class CTGAN(GenerativeModel):
    TAU = 0.2   # Gumbel-softmax temperature, as in the CTGAN paper

    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, n_classes, device, seed)
        hp = self.hp
        self.spans = info["spans"]
        self.d = int(info["shape"][0])
        # Conditional vector: one block per discrete column, plus the class label
        # when class-conditional generation is requested.
        self.blocks = list(info.get("discrete_blocks", []))
        self.cond_dim = sum(b - a for a, b in self.blocks) + self.n_classes
        self.z = int(hp["latent_dim"])
        h, self.pac = int(hp["hidden_dim"]), int(hp.get("pac", 1))
        self.G = nn.Sequential(Residual(self.z + self.cond_dim, h),
                               Residual(self.z + self.cond_dim + h, h),
                               nn.Linear(self.z + self.cond_dim + 2 * h, self.d)).to(self.device)
        self.D = VectorCritic(self.d, self.cond_dim, h, self.pac, dropout=0.5).to(self.device)

    def _activate(self, raw, hard=False):
        parts = []
        for a, b, kind in self.spans:
            if kind == "alpha":
                parts.append(torch.tanh(raw[:, a:b]))
            elif hard:
                parts.append(F.one_hot(raw[:, a:b].argmax(1), b - a).float())
            else:
                parts.append(F.gumbel_softmax(raw[:, a:b], tau=self.TAU, hard=False))
        return torch.cat(parts, 1)

    def _setup_sampler(self, X: np.ndarray, y):
        """Training-by-sampling: choose a discrete block uniformly, a category with
        probability proportional to log(1 + frequency), then a real row with it."""
        self.choices = []          # (cond offset, n categories, rows per category, log-probs)
        offset = 0
        for a, b in self.blocks:
            cats = X[:, a:b].argmax(1)
            rows = [np.flatnonzero(cats == k) for k in range(b - a)]
            freq = np.array([len(r) for r in rows], float)
            p = np.log1p(freq)
            p = p / p.sum()
            self.choices.append((offset, b - a, rows, p, freq / freq.sum()))
            offset += b - a
        if self.n_classes:
            rows = [np.flatnonzero(y == k) for k in range(self.n_classes)]
            freq = np.array([len(r) for r in rows], float)
            p = np.log1p(freq)
            self.choices.append((offset, self.n_classes, rows, p / p.sum(), freq / freq.sum()))

    def _draw(self, n, rng, train=True):
        cond = np.zeros((n, self.cond_dim), np.float32)
        if not self.choices:
            return cond, None, None
        which = rng.integers(len(self.choices), size=n)
        rows, block_of = np.zeros(n, int), np.zeros(n, int)
        for i, j in enumerate(which):
            off, k, members, p_train, p_true = self.choices[j]
            cat = rng.choice(k, p=p_train if train else p_true)
            while not len(members[cat]):
                cat = rng.choice(k, p=p_train)
            cond[i, off + cat] = 1
            rows[i] = rng.choice(members[cat])
            block_of[i] = j
        return cond, rows, which

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        Xn = np.asarray(X, np.float32)
        y = None if y is None else np.asarray(y)
        self._setup_sampler(Xn, y)
        X_t = self.tensor(Xn)
        Y_t = self.onehot(y)
        rng = np.random.default_rng(self.seed)
        lr, lam = float(self.hp["lr"]), float(self.hp["gp_lambda"])
        wd = float(self.hp.get("weight_decay", 1e-6))
        optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=wd)
        optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))
        bs = int(self.hp["batch_size"]) // self.pac * self.pac

        def cond_ce(fake_raw, cond_t):
            """Cross-entropy forcing the generated row to honour its condition."""
            loss, off = 0.0, 0
            for a, b in self.blocks:
                mask = cond_t[:, off:off + b - a].sum(1) > 0
                if mask.any():
                    loss = loss + F.cross_entropy(fake_raw[mask, a:b],
                                                  cond_t[mask, off:off + b - a].argmax(1),
                                                  reduction="sum")
                off += b - a
            return loss / cond_t.shape[0]

        def step(idx):
            n = len(idx) // self.pac * self.pac
            if n == 0:
                return {}
            cond, rows, _ = self._draw(n, rng)
            if rows is None:
                rows = idx[:n].cpu().numpy()
            cond_t = self.tensor(cond)
            real = X_t[rows]
            z = torch.randn(n, self.z, device=self.device)
            fake = self._activate(self.G(torch.cat([z, cond_t], 1))).detach()
            c_in = cond_t if self.cond_dim else None
            w = self.D(real, c_in).mean() - self.D(fake, c_in).mean()
            lossD = -w + lam * gradient_penalty(self.D, real, fake, c_in)
            optD.zero_grad()
            lossD.backward()
            optD.step()

            z = torch.randn(n, self.z, device=self.device)
            raw = self.G(torch.cat([z, cond_t], 1))
            lossG = -self.D(self._activate(raw), c_in).mean() + cond_ce(raw, cond_t)
            optG.zero_grad()
            lossG.backward()
            optG.step()
            return {"wasserstein": w.item(), "generator_loss": lossG.item()}

        self.history = train_loop(step, len(Xn), bs, int(self.hp["max_epochs"]),
                                  self.spec.train_seconds, [self.G, self.D], None,
                                  seed=self.seed, device=self.device)
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None, batch=2048):
        rng = np.random.default_rng(self.seed + 1)
        self.G.eval()
        out = []
        for i in range(0, n, batch):
            k = min(batch, n - i)
            cond = np.zeros((k, self.cond_dim), np.float32)
            if y is not None and self.n_classes:
                # Class-conditional: fix the label block.
                off = self.cond_dim - self.n_classes
                cond[np.arange(k), off + np.asarray(y)[i:i + k]] = 1
            elif self.choices:
                # Unconditional: draw conditions at their true frequencies.
                cond, _, _ = self._draw(k, rng, train=False)
            z = torch.randn(k, self.z, device=self.device)
            raw = self.G(torch.cat([z, self.tensor(cond)], 1))
            out.append(self._activate(raw, hard=True).cpu())
        return torch.cat(out).numpy()
