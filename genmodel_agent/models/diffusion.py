"""Denoising diffusion: a small U-Net for images, an MLP denoiser for vectors.

DDPM (Ho et al. 2020). The forward process gradually adds Gaussian noise,

    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I),

and a network eps_theta(x_t, t, c) is trained to predict eps with a plain MSE --
a reweighted variational bound. Choices, each a standard published one:

  * cosine noise schedule (Nichol & Dhariwal 2021): destroys information more
    evenly than the linear schedule, which matters at 28x28 where the linear one
    spends many steps on almost-pure noise;
  * deterministic DDIM sampling (Song et al. 2021) on a strided subset of
    ``sample_steps`` timesteps, so generation costs ~100 network calls, not 1000;
  * classifier-free guidance (Ho & Salimans 2022): the class label is dropped
    with probability ``p_uncond`` during training so one network learns both
    conditional and unconditional scores; at sampling
        eps = (1 + w) eps(x, c) - w eps(x, null)
    with w = ``guidance_scale``. Larger w gives more class-typical samples at the
    cost of diversity -- a trade-off the critic can act on;
  * v-prediction (Salimans & Ho 2022): the network predicts
        v = sqrt(abar) eps - sqrt(1 - abar) x0
    rather than eps. With eps-prediction the implied x0 = (x_t - sqrt(1-abar) eps)
    / sqrt(abar) divides by sqrt(abar) ~ 0.003 at the start of sampling, so any
    error is amplified ~300x; in development this left samples as coloured noise
    after 1.6k steps although partial denoising of real images worked. v is well
    conditioned at every noise level;
  * an exponential moving average of the weights (decay 0.999, warmed up) is
    used for sampling, as in the DDPM paper.

The diffusion training loss keeps falling slowly long after samples stop
improving, so it is a poor early-stopping signal; training runs to the time or
epoch cap and the validation denoising loss is recorded, not used to stop.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import GenerativeModel
from .train import train_loop


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0])[1:]
    return abar.clamp(1e-5, 0.9999).float()


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], 1)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb, c_out)
        self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(e))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm, self.attn = nn.GroupNorm(8, c), nn.MultiheadAttention(c, 4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        return x + self.attn(h, h, h, need_weights=False)[0].transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    """Two downsamplings (H -> H/2 -> H/4), self-attention at the lowest resolution."""

    def __init__(self, C, ch, n_classes):
        super().__init__()
        emb = 4 * ch
        self.ch = ch
        self.t_mlp = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.n_classes = n_classes
        self.c_emb = nn.Embedding(n_classes + 1, emb) if n_classes else None   # last = null
        self.inp = nn.Conv2d(C, ch, 3, padding=1)
        self.d1, self.d2 = ResBlock(ch, ch, emb), ResBlock(2 * ch, 2 * ch, emb)
        self.down1 = nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1)
        self.mid1, self.mid_attn = ResBlock(2 * ch, 2 * ch, emb), Attention(2 * ch)
        self.mid2 = ResBlock(2 * ch, 2 * ch, emb)
        self.u2 = ResBlock(4 * ch, 2 * ch, emb)
        self.u1 = ResBlock(3 * ch, ch, emb)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, C, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, c=None):
        e = self.t_mlp(timestep_embedding(t, self.ch))
        if self.c_emb is not None:
            e = e + self.c_emb(c)
        h0 = self.d1(self.inp(x), e)                       # H
        h1 = self.d2(self.down1(h0), e)                    # H/2
        h = self.down2(h1)                                 # H/4
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        h = F.interpolate(h, size=h1.shape[-2:], mode="nearest")
        h = self.u2(torch.cat([h, h1], 1), e)
        h = F.interpolate(h, size=h0.shape[-2:], mode="nearest")
        h = self.u1(torch.cat([h, h0], 1), e)
        return self.out(h)


class MLPDenoiser(nn.Module):
    def __init__(self, d, hidden, n_layers, n_classes):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Linear(d, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.n_classes = n_classes
        self.c_emb = nn.Embedding(n_classes + 1, hidden) if n_classes else None
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(),
                                                   nn.Linear(hidden, hidden))
                                     for _ in range(n_layers)])
        self.out = nn.Linear(hidden, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, c=None):
        e = self.t_mlp(timestep_embedding(t, self.hidden))
        if self.c_emb is not None:
            e = e + self.c_emb(c)
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h + e)                           # residual, embedding injected
        return self.out(h)


# ---------------------------------------------------------------------------
# Shared diffusion machinery
# ---------------------------------------------------------------------------

class Diffusion(GenerativeModel):
    EMA_DECAY = 0.999
    VAL_EVERY = 5

    def build_net(self) -> nn.Module:
        raise NotImplementedError

    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, n_classes, device, seed)
        self.T = int(self.hp["timesteps"])
        self.abar = cosine_alpha_bar(self.T).to(self.device)
        self.net = self.build_net().to(self.device)
        self.ema = copy.deepcopy(self.net).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.null = self.n_classes            # index of the "no label" embedding

    def _noisy(self, x0, t, eps):
        a = self.abar[t].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def _v(self, x0, t, eps):
        a = self.abar[t].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * eps - (1 - a).sqrt() * x0

    def _labels(self, y, n):
        if not self.n_classes:
            return None
        if y is None:
            return torch.full((n,), self.null, dtype=torch.long, device=self.device)
        return torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        X = self.tensor(X)
        Y = self._labels(y, len(X))
        p_uncond = float(self.hp.get("p_uncond", 0.1))
        n_updates = {"n": 0}
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.hp["lr"]),
                                weight_decay=float(self.hp.get("weight_decay", 0.0)))

        def step(idx):
            x0 = X[idx]
            t = torch.randint(0, self.T, (len(idx),), device=self.device)
            eps = torch.randn_like(x0)
            c = None
            if Y is not None:
                c = Y[idx].clone()
                c[torch.rand(len(idx), device=self.device) < p_uncond] = self.null
            loss = F.mse_loss(self.net(self._noisy(x0, t, eps), t, c), self._v(x0, t, eps))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            # EMA with warm-up: a fixed 0.999 decay would leave the average close
            # to the initial weights for the first few thousand steps (observed:
            # pure-noise samples after 340 steps), so the decay ramps up instead.
            n_updates["n"] += 1
            decay = min(self.EMA_DECAY, (1 + n_updates["n"]) / (10 + n_updates["n"]))
            with torch.no_grad():
                for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                    pe.lerp_(p, 1 - decay)
            return {"loss": loss.item()}

        # Validation denoising loss with fixed (t, eps), so successive values are comparable.
        val = None
        if X_val is not None and len(X_val):
            g = torch.Generator().manual_seed(self.seed)
            Xv = self.tensor(X_val)[:2048]
            val = (Xv, torch.randint(0, self.T, (len(Xv),), generator=g).to(self.device),
                   torch.randn(Xv.shape, generator=g).to(self.device),
                   None if y_val is None else self._labels(np.asarray(y_val)[:2048], len(Xv)))
        val_curve = []

        def record_val(epoch):
            if val is not None and (epoch == 1 or epoch % self.VAL_EVERY == 0):
                with torch.no_grad():
                    Xv, tv, ev, cv = val
                    loss = F.mse_loss(self.ema(self._noisy(Xv, tv, ev), tv, cv),
                                      self._v(Xv, tv, ev)).item()
                val_curve.append({"epoch": epoch, "val_loss": loss})

        self.history = train_loop(step, len(X), int(self.hp["batch_size"]),
                                  int(self.hp["max_epochs"]), self.spec.train_seconds,
                                  [self.net], None, seed=self.seed, device=self.device,
                                  on_epoch_end=record_val)
        self.history["val_curve"] = val_curve
        self.history["optimizer_steps"] = n_updates["n"]
        self.history["best_val_loss"] = val_curve[-1]["val_loss"] if val_curve else None
        return self.history

    @torch.no_grad()
    def _ddim(self, n, y, shape, guidance):
        steps = int(min(self.hp["sample_steps"], self.T))
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).round().long()
        g = torch.Generator().manual_seed(self.seed + 1)
        x = torch.randn((n, *shape), generator=g).to(self.device)
        c = self._labels(y, n)
        use_cfg = c is not None and y is not None and guidance > 0
        null = torch.full_like(c, self.null) if use_cfg else None
        for i, t in enumerate(ts):
            tt = torch.full((n,), int(t), device=self.device, dtype=torch.long)
            if use_cfg:
                v = self.ema(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([c, null]))
                v_c, v_u = v.chunk(2)
                v = (1 + guidance) * v_c - guidance * v_u      # guidance is linear in v
            else:
                v = self.ema(x, tt, c)
            a_t = self.abar[t]
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            x0 = (a_t.sqrt() * x - (1 - a_t).sqrt() * v).clamp(*self.clip_range)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp_min(1e-4)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x

    clip_range = (-1e6, 1e6)

    @torch.no_grad()
    def sample(self, n, y=None, batch=None):
        batch = batch or self.sample_batch
        guidance = float(self.hp.get("guidance_scale", 0.0))
        y_arr = None if y is None else np.asarray(y)
        out = []
        for i in range(0, n, batch):
            k = min(batch, n - i)
            out.append(self._ddim(k, None if y_arr is None else y_arr[i:i + k],
                                  self.sample_shape, guidance).cpu())
        return torch.cat(out).numpy()


class ImageDDPM(Diffusion):
    clip_range = (-1.0, 1.0)      # predicted x0 is clipped to the pixel range
    sample_batch = 500

    def build_net(self):
        C, H, W = self.info["shape"]
        self.sample_shape = (C, H, W)
        return UNet(C, int(self.hp["base_channels"]), self.n_classes)


class VectorDDPM(Diffusion):
    clip_range = (-6.0, 6.0)      # data are standardised; |x0| > 6 sd is an artefact
    sample_batch = 4096

    def build_net(self):
        d = int(self.info["shape"][0])
        self.sample_shape = (d,)
        return MLPDenoiser(d, int(self.hp["hidden_dim"]), int(self.hp["n_layers"]),
                           self.n_classes)
