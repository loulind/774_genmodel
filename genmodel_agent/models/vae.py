"""Variational autoencoders: Gaussian/categorical, count (NB/ZINB), and masked (MIWAE).

All three maximise an evidence lower bound

    log p(x) >= E_q[log p(x | z)] - beta * KL(q(z | x) || N(0, I))

and differ only in the likelihood p(x | z), which is where the data type enters:

  * VAE        Gaussian with a *learned* noise scale for continuous values (the
               sigma-VAE of Rybkin et al. 2021; a fixed sigma silently sets the
               weight of reconstruction against KL and is a common cause of
               blurry samples), softmax/categorical for one-hot blocks.
  * CountVAE   negative binomial or zero-inflated NB on raw counts, with mean
               library_size * softmax(decoder) and gene-wise dispersion, as in
               scVI (Lopez et al. 2018). The observed library size is an offset,
               so the latent space describes composition, not sequencing depth.
  * MaskedVAE  MIWAE (Mattei & Frellsen 2019): the likelihood is evaluated on
               observed entries only and the bound is importance-weighted, so it
               trains on incomplete rows and imputes by self-normalised
               importance sampling.

Conditioning (when ``spec.conditional``) concatenates a one-hot label to the
encoder input and the decoder input: a conditional VAE (Sohn et al. 2015).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import GenerativeModel
from .train import train_loop


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def mlp(d_in: int, hidden: int, n_layers: int, d_out: int, dropout: float = 0.0,
        norm: bool = False) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden)]
        if norm:
            layers += [nn.BatchNorm1d(hidden)]
        layers += [nn.ReLU()]
        if dropout:
            layers += [nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    """(C, H, W) -> features; two stride-2 convolutions (H, W divisible by 4)."""

    def __init__(self, C, H, W, ch, d_cond, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.BatchNorm2d(2 * ch), nn.LeakyReLU(0.2),
            nn.Flatten())
        self.head = nn.Sequential(nn.Linear(2 * ch * (H // 4) * (W // 4) + d_cond, 256),
                                  nn.LeakyReLU(0.2), nn.Linear(256, d_out))

    def forward(self, x, c=None):
        h = self.net(x)
        return self.head(h if c is None else torch.cat([h, c], 1))


class ConvDecoder(nn.Module):
    def __init__(self, C, H, W, ch, d_in):
        super().__init__()
        self.shape = (2 * ch, H // 4, W // 4)
        self.fc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                nn.Linear(256, int(np.prod(self.shape))), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, C, 4, 2, 1))

    def forward(self, z):
        return self.net(self.fc(z).view(-1, *self.shape))


def kl_normal(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)


def reparameterise(mu, logvar):
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


def split_spans(spans, dim):
    """Indices of continuous columns and list of categorical (start, end) blocks."""
    if not spans:
        return torch.arange(dim), []
    cont = [a for a, b, k in spans if k == "cont"]
    cats = [(a, b) for a, b, k in spans if k == "cat"]
    return torch.tensor(cont, dtype=torch.long), cats


def mixed_loglik(x, out, log_sigma, cont_idx, cats, mask=None):
    """Per-sample log p(x | z): Gaussian on continuous columns, categorical on blocks.

    ``mask`` (same shape as x, 1 = observed) restricts the sum to observed entries.
    """
    ll = torch.zeros(x.shape[0], device=x.device)
    if len(cont_idx):
        xc, mc = x[:, cont_idx], out[:, cont_idx]
        ls = log_sigma[cont_idx] if log_sigma.dim() == 1 else log_sigma[:, cont_idx]
        g = -0.5 * ((xc - mc) / ls.exp()).pow(2) - ls - 0.5 * math.log(2 * math.pi)
        if mask is not None:
            g = g * mask[:, cont_idx]
        ll = ll + g.sum(1)
    for a, b in cats:
        logp = F.log_softmax(out[:, a:b], dim=1)
        c = (x[:, a:b] * logp).sum(1)
        if mask is not None:
            c = c * mask[:, a]
        ll = ll + c
    return ll


def sample_mixed(out, log_sigma, cont_idx, cats, noise=True):
    x = out.clone()
    if len(cont_idx) and noise:
        ls = log_sigma[cont_idx] if log_sigma.dim() == 1 else log_sigma[:, cont_idx]
        x[:, cont_idx] = out[:, cont_idx] + ls.exp() * torch.randn_like(out[:, cont_idx])
    for a, b in cats:
        idx = torch.multinomial(F.softmax(out[:, a:b], 1), 1).squeeze(1)
        x[:, a:b] = F.one_hot(idx, b - a).float()
    return x


# ---------------------------------------------------------------------------
# Gaussian / categorical (conditional) VAE
# ---------------------------------------------------------------------------

class VAE(GenerativeModel):
    supports_score = True
    supports_encode = True

    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, n_classes, device, seed)
        hp = self.hp
        self.latent = int(hp["latent_dim"])
        self.beta = float(hp.get("beta", 1.0))
        c = self.n_classes
        self.is_image = info["kind"] == "image" and info["shape"][1] % 4 == 0 \
            and info["shape"][2] % 4 == 0
        if self.is_image:
            C, H, W = info["shape"]
            ch = max(16, int(hp["hidden_dim"]) // 4)
            self.enc = ConvEncoder(C, H, W, ch, c, 2 * self.latent).to(self.device)
            self.dec = ConvDecoder(C, H, W, ch, self.latent + c).to(self.device)
            # One learned noise scale per channel (sigma-VAE).
            self.log_sigma = nn.Parameter(torch.zeros(C, 1, 1, device=self.device))
            self.dim = C * H * W
            self.cont_idx, self.cats = None, []
        else:
            d = int(np.prod(info["shape"]))
            h, L = int(hp["hidden_dim"]), int(hp.get("n_layers", 2))
            self.enc = mlp(d + c, h, L, 2 * self.latent).to(self.device)
            self.dec = mlp(self.latent + c, h, L, d).to(self.device)
            self.log_sigma = nn.Parameter(torch.zeros(d, device=self.device))
            self.dim = d
            cont, self.cats = split_spans(info.get("spans"), d)
            self.cont_idx = cont.to(self.device)
        self.params = nn.ParameterList([self.log_sigma])

    def _encode(self, x, c):
        if self.is_image:
            h = self.enc(x, c)
        else:
            h = self.enc(x if c is None else torch.cat([x, c], 1))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-10, 6)

    def _decode(self, z, c):
        return self.dec(z if c is None else torch.cat([z, c], 1))

    def _loglik(self, x, out):
        ls = self.log_sigma.clamp(-6, 2)
        if self.is_image:
            g = -0.5 * ((x - out) / ls.exp()).pow(2) - ls - 0.5 * math.log(2 * math.pi)
            return g.flatten(1).sum(1)
        return mixed_loglik(x, out, ls, self.cont_idx, self.cats)

    def elbo(self, x, c, beta=None):
        mu, logvar = self._encode(x, c)
        out = self._decode(reparameterise(mu, logvar), c)
        rec = self._loglik(x, out)
        kl = kl_normal(mu, logvar)
        return rec - (self.beta if beta is None else beta) * kl, rec, kl

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        X, Xv = self.tensor(X), None if X_val is None else self.tensor(X_val)
        C, Cv = self.onehot(y), self.onehot(y_val)
        params = list(self.enc.parameters()) + list(self.dec.parameters()) + [self.log_sigma]
        opt = torch.optim.Adam(params, lr=float(self.hp["lr"]),
                               weight_decay=float(self.hp.get("weight_decay", 0.0)))

        def step(idx):
            elbo, rec, kl = self.elbo(X[idx], None if C is None else C[idx])
            loss = -elbo.mean() / self.dim
            opt.zero_grad()
            loss.backward()
            opt.step()
            return {"loss": -elbo.mean().item(), "rec": -rec.mean().item(), "kl": kl.mean().item()}

        def validate():
            if Xv is None:
                return None
            return float(-self.elbo(Xv, Cv, beta=self.beta)[0].mean())

        self.history = train_loop(step, len(X), int(self.hp["batch_size"]),
                                  int(self.hp["max_epochs"]), self.spec.train_seconds,
                                  [self.enc, self.dec, self.params], validate,
                                  seed=self.seed, device=self.device)
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None, batch=2048):
        out = []
        for i in range(0, n, batch):
            k = min(batch, n - i)
            z = torch.randn(k, self.latent, device=self.device)
            c = self.onehot(None if y is None else np.asarray(y)[i:i + k])
            m = self._decode(z, c)
            if self.is_image:
                # Image samples are decoder means: per-pixel noise would add only
                # salt-and-pepper texture, which is not how images vary.
                out.append(m.clamp(-1, 1).cpu())
            else:
                out.append(sample_mixed(m, self.log_sigma.clamp(-6, 2), self.cont_idx,
                                        self.cats).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def score(self, X, y=None, extras=None, n_draws=5):
        X, C = self.tensor(X), self.onehot(y)
        return np.mean([self.elbo(X, C)[0].cpu().numpy() for _ in range(n_draws)], axis=0)

    @torch.no_grad()
    def encode(self, X, y=None):
        return self._encode(self.tensor(X), self.onehot(y))[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Count VAE (NB / ZINB), scVI-style
# ---------------------------------------------------------------------------

def nb_loglik(x, mu, log_theta):
    theta = log_theta.exp()
    log_mu = torch.log(mu + 1e-8)
    log_theta_mu = torch.logaddexp(log_theta, log_mu)
    return (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (log_theta - log_theta_mu) + x * (log_mu - log_theta_mu))


def zinb_loglik(x, mu, log_theta, pi_logits):
    nb = nb_loglik(x, mu, log_theta)
    log_pi = F.logsigmoid(pi_logits)
    log_1mpi = F.logsigmoid(-pi_logits)
    zero = torch.logaddexp(log_pi, log_1mpi + nb)
    return torch.where(x < 0.5, zero, log_1mpi + nb)


class CountVAE(GenerativeModel):
    supports_score = True
    supports_encode = True

    KL_WARMUP_EPOCHS = 50   # linear KL annealing, as scVI does, to avoid posterior collapse

    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, n_classes, device, seed)
        hp = self.hp
        self.zero_inflated = spec.name == "zinb_vae"
        G = int(info["shape"][0])
        self.G, self.latent = G, int(hp["latent_dim"])
        c, h, L = self.n_classes, int(hp["hidden_dim"]), int(hp.get("n_layers", 1))
        self.enc = mlp(G + c, h, L, 2 * self.latent, dropout=0.1, norm=True).to(self.device)
        self.dec_body = mlp(self.latent + c, h, L, h, norm=True).to(self.device)
        self.dec_scale = nn.Linear(h, G).to(self.device)
        self.dec_pi = nn.Linear(h, G).to(self.device)
        self.log_theta = nn.Parameter(torch.zeros(G, device=self.device))
        self.params = nn.ParameterList([self.log_theta])
        self.beta = float(hp.get("beta", 1.0))
        self.kl_weight = 0.0

    def _encode(self, x, c):
        h = torch.log1p(x)
        h = self.enc(h if c is None else torch.cat([h, c], 1))
        mu, logvar = h.chunk(2, 1)
        return mu, logvar.clamp(-10, 6)

    def _decode(self, z, c):
        h = F.relu(self.dec_body(z if c is None else torch.cat([z, c], 1)))
        return F.softmax(self.dec_scale(h), dim=1), self.dec_pi(h)

    def _loglik(self, x, lib, scale, pi_logits):
        mu = lib[:, None] * scale
        lt = self.log_theta.clamp(-8, 12)
        if self.zero_inflated:
            return zinb_loglik(x, mu, lt, pi_logits).sum(1)
        return nb_loglik(x, mu, lt).sum(1)

    def elbo(self, x, lib, c, beta):
        mu, logvar = self._encode(x, c)
        scale, pi = self._decode(reparameterise(mu, logvar), c)
        rec = self._loglik(x, lib, scale, pi)
        kl = kl_normal(mu, logvar)
        return rec - beta * kl, rec, kl

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        X = self.tensor(X)
        lib = self.tensor(extras["library"])
        C = self.onehot(y)
        Xv = None if X_val is None else self.tensor(X_val)
        libv = None if X_val is None else self.tensor(extras_val["library"])
        Cv = self.onehot(y_val)
        # Library sizes are resampled per class at generation time.
        self.library_by_class = {}
        y_arr = np.zeros(len(X), int) if y is None else np.asarray(y)
        for k in np.unique(y_arr):
            self.library_by_class[int(k)] = np.asarray(extras["library"])[y_arr == k]
        self.library_all = np.asarray(extras["library"])

        modules = [self.enc, self.dec_body, self.dec_scale, self.dec_pi, self.params]
        opt = torch.optim.Adam([p for m in modules for p in m.parameters()],
                               lr=float(self.hp["lr"]),
                               weight_decay=float(self.hp.get("weight_decay", 0.0)), eps=0.01)

        def step(idx):
            elbo, rec, kl = self.elbo(X[idx], lib[idx], None if C is None else C[idx],
                                      self.kl_weight * self.beta)
            loss = -elbo.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            return {"loss": (-(rec - self.beta * kl).mean()).item(), "rec": -rec.mean().item(),
                    "kl": kl.mean().item()}

        def validate():
            if Xv is None or self.kl_weight < 1.0:
                return None          # do not select a checkpoint during KL warm-up
            return float(-self.elbo(Xv, libv, Cv, self.beta)[0].mean())

        def anneal(epoch):
            self.kl_weight = min(1.0, epoch / self.KL_WARMUP_EPOCHS)

        self.history = train_loop(step, len(X), int(self.hp["batch_size"]),
                                  int(self.hp["max_epochs"]), self.spec.train_seconds,
                                  modules, validate, seed=self.seed, device=self.device,
                                  on_epoch_end=anneal)
        self.history["kl_warmup_epochs"] = self.KL_WARMUP_EPOCHS
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None, batch=1024):
        rng = np.random.default_rng(self.seed)
        y_arr = None if y is None else np.asarray(y)
        out = []
        for i in range(0, n, batch):
            k = min(batch, n - i)
            yk = None if y_arr is None else y_arr[i:i + k]
            if yk is not None and self.n_classes:
                lib = np.array([rng.choice(self.library_by_class.get(int(c), self.library_all))
                                for c in yk])
            else:
                lib = rng.choice(self.library_all, k)
            z = torch.randn(k, self.latent, device=self.device)
            scale, pi = self._decode(z, self.onehot(yk))
            mu = (torch.as_tensor(lib, dtype=torch.float32, device=self.device)[:, None]
                  * scale).cpu().double()
            theta = self.log_theta.clamp(-8, 12).exp().cpu().double()
            # Gamma-Poisson mixture = NB(mu, theta); sampled on CPU (no MPS kernels).
            lam = torch.distributions.Gamma(theta.expand_as(mu), theta / mu.clamp_min(1e-8)).sample()
            x = torch.poisson(lam)
            if self.zero_inflated:
                drop = torch.bernoulli(torch.sigmoid(pi.cpu().double()))
                x = x * (1 - drop)
            out.append(x.float())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def score(self, X, y=None, extras=None):
        X = self.tensor(X)
        lib = self.tensor(extras["library"]) if extras else X.sum(1)
        return self.elbo(X, lib, self.onehot(y), self.beta)[0].cpu().numpy()

    @torch.no_grad()
    def encode(self, X, y=None):
        return self._encode(self.tensor(X), self.onehot(y))[0].cpu().numpy()

    def summary(self):
        s = super().summary()
        theta = self.log_theta.detach().clamp(-8, 12).exp().cpu().numpy()
        s["median_inverse_dispersion_theta"] = float(np.median(theta))
        return s


# ---------------------------------------------------------------------------
# MIWAE: masked VAE for imputation
# ---------------------------------------------------------------------------

class MaskedVAE(GenerativeModel):
    supports_score = True
    supports_impute = True

    def __init__(self, spec, info, n_classes=0, device="cpu", seed=0):
        super().__init__(spec, info, 0, device, seed)
        hp = self.hp
        d = int(info["shape"][0])
        self.d, self.latent, self.K = d, int(hp["latent_dim"]), int(hp["n_importance"])
        h, L = int(hp["hidden_dim"]), int(hp.get("n_layers", 2))
        # The encoder sees the zero-filled data and the mask, so it can tell
        # "observed zero" from "missing".
        self.enc = mlp(2 * d, h, L, 2 * self.latent).to(self.device)
        self.dec = mlp(self.latent, h, L, 2 * d).to(self.device)   # mean and log-sigma
        cont, self.cats = split_spans(info.get("spans"), d)
        self.cont_idx = cont.to(self.device)

    def _log_weights(self, x, m, K):
        """log w_k = log p(x_obs|z_k) + log p(z_k) - log q(z_k|x), shape (K, n)."""
        h = self.enc(torch.cat([x * m, m], 1))
        mu, logvar = h.chunk(2, 1)
        logvar = logvar.clamp(-10, 6)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn(K, *mu.shape, device=x.device)        # (K, n, L)
        out = self.dec(z.reshape(-1, self.latent))
        mean, log_sigma = out.chunk(2, 1)
        log_sigma = log_sigma.clamp(-6, 2)
        xr, mr = x.repeat(K, 1), m.repeat(K, 1)
        ll = mixed_loglik(xr, mean, log_sigma, self.cont_idx, self.cats, mr).view(K, -1)
        log_pz = (-0.5 * z.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_qz = (-0.5 * ((z - mu) / std).pow(2) - 0.5 * logvar
                  - 0.5 * math.log(2 * math.pi)).sum(-1)
        return ll + log_pz - log_qz, mean.view(K, -1, self.d), log_sigma.view(K, -1, self.d)

    def bound(self, x, m, K):
        lw, _, _ = self._log_weights(x, m, K)
        return torch.logsumexp(lw, 0) - math.log(K)

    def fit(self, X, y=None, X_val=None, y_val=None, extras=None, extras_val=None):
        X, M = self.tensor(X), self.tensor(extras["mask"])
        Xv = None if X_val is None else self.tensor(X_val)
        Mv = None if X_val is None else self.tensor(extras_val["mask"])
        modules = [self.enc, self.dec]
        opt = torch.optim.Adam([p for m in modules for p in m.parameters()],
                               lr=float(self.hp["lr"]),
                               weight_decay=float(self.hp.get("weight_decay", 0.0)))

        def step(idx):
            loss = -self.bound(X[idx], M[idx], self.K).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            return {"loss": loss.item()}

        def validate():
            return None if Xv is None else float(-self.bound(Xv, Mv, self.K).mean())

        self.history = train_loop(step, len(X), int(self.hp["batch_size"]),
                                  int(self.hp["max_epochs"]), self.spec.train_seconds,
                                  modules, validate, patience=50, seed=self.seed,
                                  device=self.device)
        return self.history

    @torch.no_grad()
    def impute(self, X, mask, n_draws=1, K=200):
        """n_draws=1: importance-weighted conditional mean (single imputation).
        n_draws>1: sampling-importance-resampling draws (multiple imputation),
        returned with shape (n_draws, n, d)."""
        X, M = self.tensor(X), self.tensor(mask)
        lw, mean, log_sigma = self._log_weights(X, M, K)
        w = torch.softmax(lw, 0)                                         # (K, n)
        if n_draws == 1:
            filled = mean.clone()
            for a, b in self.cats:
                filled[:, :, a:b] = torch.softmax(mean[:, :, a:b], -1)
            est = (w[:, :, None] * filled).sum(0)
            for a, b in self.cats:
                est[:, a:b] = F.one_hot(est[:, a:b].argmax(1), b - a).float()
            return (X * M + est * (1 - M)).cpu().numpy()
        draws = []
        for _ in range(n_draws):
            k = torch.multinomial(w.T, 1).squeeze(1)                      # (n,)
            rows = torch.arange(X.shape[0], device=X.device)
            x = sample_mixed(mean[k, rows], log_sigma[k, rows], self.cont_idx, self.cats)
            draws.append((X * M + x * (1 - M)).cpu().numpy())
        return np.stack(draws)

    @torch.no_grad()
    def score(self, X, y=None, extras=None):
        mask = extras["mask"] if extras else np.ones_like(X)
        return self.bound(self.tensor(X), self.tensor(mask), self.K).cpu().numpy()

    @torch.no_grad()
    def sample(self, n, y=None):
        z = torch.randn(n, self.latent, device=self.device)
        mean, log_sigma = self.dec(z).chunk(2, 1)
        return sample_mixed(mean, log_sigma.clamp(-6, 2), self.cont_idx, self.cats).cpu().numpy()
