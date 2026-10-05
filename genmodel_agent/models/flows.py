"""Normalizing flows: RealNVP and MAF, with exact log-likelihoods.

A flow is an invertible map f from data x to a standard-normal z, so by change
of variables

    log p(x) = log N(f(x); 0, I) + log |det df/dx|.

Each layer is designed so the Jacobian determinant is cheap:
  * RealNVP (Dinh et al. 2017): affine coupling -- half the coordinates are
    scaled and shifted by functions of the other half; the Jacobian is
    triangular, so log|det| is the sum of the log-scales. Fast both ways.
  * MAF (Papamakarios et al. 2017): each coordinate is an affine function of the
    *preceding* coordinates, computed in one pass by a masked network (MADE,
    Germain et al. 2015). Density evaluation is one pass; sampling is
    sequential over dimensions -- a good trade for anomaly scoring, which only
    needs densities.
ActNorm layers (Kingma & Dhariwal 2018) between blocks standardise activations
with data-initialised affine parameters, which stabilises training.

Dequantisation: a continuous density on data with duplicated rows is degenerate
-- the likelihood can grow without bound by concentrating mass on the repeats.
When the planner sees duplicates it sets ``noise_std`` > 0 and the flow is
trained on x + N(0, noise_std^2) (in standardised units).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import GenerativeModel
from .train import train_loop


class ActNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(d))
        self.log_scale = nn.Parameter(torch.zeros(d))
        self.register_buffer("initialised", torch.tensor(False))

    def forward(self, x):
        if not self.initialised:
            with torch.no_grad():
                self.loc.data = -x.mean(0)
                self.log_scale.data = -torch.log(x.std(0) + 1e-6)
                self.initialised.fill_(True)
        y = (x + self.loc) * self.log_scale.exp()
        return y, self.log_scale.sum().expand(x.shape[0])

    def inverse(self, y):
        return y * (-self.log_scale).exp() - self.loc


class AffineCoupling(nn.Module):
    def __init__(self, d, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 2 * d))
        nn.init.zeros_(self.net[-1].weight)        # start as the identity map
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x_masked):
        s, t = self.net(x_masked).chunk(2, 1)
        s = torch.tanh(s) * (1 - self.mask)
        return s, t * (1 - self.mask)

    def forward(self, x):
        s, t = self._st(x * self.mask)
        return x * self.mask + (1 - self.mask) * (x * s.exp() + t), s.sum(1)

    def inverse(self, y):
        s, t = self._st(y * self.mask)
        return y * self.mask + (1 - self.mask) * ((y - t) * (-s).exp())


class MaskedLinear(nn.Linear):
    def __init__(self, d_in, d_out, mask):
        super().__init__(d_in, d_out)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Autoregressive layer: output i depends only on inputs < i (in self.order)."""

    def __init__(self, d, hidden, order):
        super().__init__()
        self.register_buffer("order", order)
        rng = np.random.default_rng(int(order[0]))
        deg_in = order.numpy()
        deg_h1 = rng.integers(0, max(d - 1, 1), hidden)
        deg_h2 = rng.integers(0, max(d - 1, 1), hidden)
        m1 = torch.tensor(deg_h1[:, None] >= deg_in[None, :], dtype=torch.float32)
        m2 = torch.tensor(deg_h2[:, None] >= deg_h1[None, :], dtype=torch.float32)
        m3 = torch.tensor(deg_in[:, None] > deg_h2[None, :], dtype=torch.float32)
        self.l1 = MaskedLinear(d, hidden, m1)
        self.l2 = MaskedLinear(hidden, hidden, m2)
        self.mu = MaskedLinear(hidden, d, m3)
        self.alpha = MaskedLinear(hidden, d, m3)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def _params(self, x):
        h = F.relu(self.l2(F.relu(self.l1(x))))
        return self.mu(h), self.alpha(h).clamp(-7, 7)

    def forward(self, x):
        mu, alpha = self._params(x)
        return (x - mu) * (-alpha).exp(), -alpha.sum(1)

    def inverse(self, u):
        x = torch.zeros_like(u)
        for i in torch.argsort(self.order):          # one dimension at a time
            mu, alpha = self._params(x)
            x[:, i] = u[:, i] * alpha[:, i].exp() + mu[:, i]
        return x


class Flow(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x):
        z, logdet = self(x)
        return (-0.5 * z.pow(2) - 0.5 * math.log(2 * math.pi)).sum(1) + logdet

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


class FlowModel(GenerativeModel):
    supports_score = True

    def build_layers(self, d, hidden, n_blocks):
        raise NotImplementedError

    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, 0, device, seed)
        d = int(info["shape"][0])
        self.d = d
        self.flow = Flow(self.build_layers(d, int(self.hp["hidden_dim"]),
                                           int(self.hp["n_blocks"]))).to(self.device)

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        X = self.tensor(X)
        Xv = None if X_val is None else self.tensor(X_val)
        noise = float(self.hp.get("noise_std", 0.0))
        with torch.no_grad():                        # data-dependent ActNorm init
            self.flow(X[torch.randperm(len(X))[:min(len(X), 1024)]])
        opt = torch.optim.Adam(self.flow.parameters(), lr=float(self.hp["lr"]),
                               weight_decay=float(self.hp.get("weight_decay", 0.0)))

        def step(idx):
            x = X[idx]
            if noise:
                x = x + noise * torch.randn_like(x)
            loss = -self.flow.log_prob(x).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
            opt.step()
            return {"loss": loss.item()}

        # Fixed validation jitter: fresh noise at every check would make the
        # validation loss itself noisy and trigger early stopping spuriously.
        Xv_noisy = None
        if Xv is not None:
            g = torch.Generator().manual_seed(self.seed)
            Xv_noisy = Xv + noise * torch.randn(Xv.shape, generator=g).to(self.device) \
                if noise else Xv

        def validate():
            return None if Xv_noisy is None else float(-self.flow.log_prob(Xv_noisy).mean())

        self.history = train_loop(step, len(X), int(self.hp["batch_size"]),
                                  int(self.hp["max_epochs"]), self.spec.train_seconds,
                                  [self.flow], validate, seed=self.seed, device=self.device)
        return self.history

    @torch.no_grad()
    def score(self, X, y=None, extras=None):
        return self.flow.log_prob(self.tensor(X)).cpu().numpy()

    @torch.no_grad()
    def sample(self, n, y=None):
        z = torch.randn(n, self.d, device=self.device)
        return self.flow.inverse(z).cpu().numpy()


class RealNVP(FlowModel):
    def build_layers(self, d, hidden, n_blocks):
        even = (torch.arange(d) % 2 == 0).float()
        layers = []
        for _ in range(n_blocks):
            layers += [ActNorm(d), AffineCoupling(d, hidden, even),
                       AffineCoupling(d, hidden, 1 - even)]
        return layers


class MAF(FlowModel):
    def build_layers(self, d, hidden, n_blocks):
        layers = []
        for b in range(n_blocks):
            order = torch.arange(d) if b % 2 == 0 else torch.arange(d).flip(0)
            layers += [ActNorm(d), MADE(d, hidden, order)]
        return layers
