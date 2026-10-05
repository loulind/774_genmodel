"""The model library as data: what exists, where it applies, and its legal ranges.

This file is the guardrail the validator enforces. A plan -- from the LLM or the
heuristic planner -- may only name models listed here, only for the modalities
and tasks listed, and only with hyperparameters inside these ranges. The
implementation classes live in sibling modules and are looked up lazily by
``build()``, so the planner and validator never import torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (low, high) inclusive. Integers stay integers when clipped.
Range = tuple[float, float]

COMMON_TRAINING: dict[str, Range] = {
    "lr": (1e-5, 1e-2),
    "batch_size": (16, 1024),
    "max_epochs": (1, 2000),
    "weight_decay": (0.0, 1e-2),
}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    family: str                     # vae | gan | flow | diffusion
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    conditional_capable: bool
    hp_ranges: dict[str, Range]
    defaults: dict[str, float | int | str | bool]
    representations: tuple[str, ...]        # data spaces the model can be trained in
    module: str                             # implementation: genmodel_agent.models.<module>
    cls: str
    summary: str
    has_likelihood: bool = False            # exact or bounded log p(x) for scoring
    notes: tuple[str, ...] = field(default_factory=tuple)


GENERATIVE_TASKS = ("generation", "augmentation", "class_balancing")

REGISTRY: dict[str, ModelInfo] = {m.name: m for m in [
    ModelInfo(
        name="vae", family="vae", modalities=("image", "tabular", "counts"),
        tasks=GENERATIVE_TASKS + ("representation_learning", "anomaly_detection"),
        conditional_capable=True,
        hp_ranges={"latent_dim": (2, 256), "hidden_dim": (16, 1024), "n_layers": (1, 4),
                   "beta": (0.05, 4.0), **COMMON_TRAINING},
        defaults={"latent_dim": 16, "hidden_dim": 256, "n_layers": 2, "beta": 1.0,
                  "lr": 1e-3, "batch_size": 128, "max_epochs": 200, "weight_decay": 0.0},
        representations=("pixels", "standardized", "lognorm_hvg"),
        module="vae", cls="VAE",
        summary="Gaussian/Bernoulli-likelihood (conditional) VAE; convolutional for images",
        has_likelihood=True,
    ),
    ModelInfo(
        name="nb_vae", family="vae", modalities=("counts",),
        tasks=GENERATIVE_TASKS + ("representation_learning",),
        conditional_capable=True,
        hp_ranges={"latent_dim": (2, 64), "hidden_dim": (32, 1024), "n_layers": (1, 3),
                   "beta": (0.05, 4.0), **COMMON_TRAINING},
        defaults={"latent_dim": 10, "hidden_dim": 128, "n_layers": 1, "beta": 1.0,
                  "lr": 1e-3, "batch_size": 128, "max_epochs": 400, "weight_decay": 1e-6},
        representations=("counts_hvg",),
        module="vae", cls="CountVAE",
        summary="scVI-style VAE with negative-binomial likelihood and observed library size",
        has_likelihood=True,
    ),
    ModelInfo(
        name="zinb_vae", family="vae", modalities=("counts",),
        tasks=GENERATIVE_TASKS + ("representation_learning",),
        conditional_capable=True,
        hp_ranges={"latent_dim": (2, 64), "hidden_dim": (32, 1024), "n_layers": (1, 3),
                   "beta": (0.05, 4.0), **COMMON_TRAINING},
        defaults={"latent_dim": 10, "hidden_dim": 128, "n_layers": 1, "beta": 1.0,
                  "lr": 1e-3, "batch_size": 128, "max_epochs": 400, "weight_decay": 1e-6},
        representations=("counts_hvg",),
        module="vae", cls="CountVAE",
        summary="as nb_vae, with a per-gene zero-inflation component (ZINB)",
        has_likelihood=True,
    ),
    ModelInfo(
        name="masked_vae", family="vae", modalities=("tabular",),
        tasks=("imputation", "uncertainty_quantification"),
        conditional_capable=False,
        hp_ranges={"latent_dim": (2, 64), "hidden_dim": (16, 512), "n_layers": (1, 3),
                   "n_importance": (1, 50), **COMMON_TRAINING},
        defaults={"latent_dim": 8, "hidden_dim": 128, "n_layers": 2, "n_importance": 10,
                  "lr": 1e-3, "batch_size": 64, "max_epochs": 500, "weight_decay": 1e-5},
        representations=("standardized_onehot",),
        module="vae", cls="MaskedVAE",
        summary="MIWAE-style VAE trained on observed entries only; imputes by importance-weighted draws",
        has_likelihood=True,
    ),
    ModelInfo(
        name="wgan_gp", family="gan", modalities=("image", "tabular", "counts"),
        tasks=GENERATIVE_TASKS, conditional_capable=True,
        hp_ranges={"latent_dim": (2, 256), "hidden_dim": (16, 1024), "gp_lambda": (0.1, 50.0),
                   "n_critic": (1, 10), **COMMON_TRAINING},
        defaults={"latent_dim": 64, "hidden_dim": 256, "gp_lambda": 10.0, "n_critic": 5,
                  "lr": 1e-4, "batch_size": 128, "max_epochs": 300, "weight_decay": 0.0},
        representations=("pixels", "standardized", "lognorm_hvg"),
        module="gan", cls="WGANGP",
        summary="Wasserstein GAN with gradient penalty; conditional via label embedding",
    ),
    ModelInfo(
        name="ctgan", family="gan", modalities=("tabular",),
        tasks=GENERATIVE_TASKS, conditional_capable=True,
        hp_ranges={"latent_dim": (8, 256), "hidden_dim": (32, 1024), "gp_lambda": (0.1, 50.0),
                   "pac": (1, 10), "max_modes": (1, 10), **COMMON_TRAINING},
        defaults={"latent_dim": 128, "hidden_dim": 256, "gp_lambda": 10.0, "pac": 1,
                  "max_modes": 5, "lr": 2e-4, "batch_size": 128, "max_epochs": 300,
                  "weight_decay": 1e-6},
        representations=("mode_specific",),
        module="gan", cls="CTGAN",
        summary="CTGAN-style: mode-specific normalisation, conditional vector, training-by-sampling",
    ),
    ModelInfo(
        name="realnvp", family="flow", modalities=("tabular",),
        tasks=GENERATIVE_TASKS + ("anomaly_detection",), conditional_capable=False,
        hp_ranges={"n_blocks": (2, 16), "hidden_dim": (16, 512), **COMMON_TRAINING},
        defaults={"n_blocks": 6, "hidden_dim": 128, "lr": 1e-3, "batch_size": 256,
                  "max_epochs": 300, "weight_decay": 1e-5},
        representations=("standardized",),
        module="flows", cls="RealNVP",
        summary="RealNVP affine coupling flow; exact log-likelihood",
        has_likelihood=True, notes=("ambient dimension must be modest (<= 200)",),
    ),
    ModelInfo(
        name="maf", family="flow", modalities=("tabular",),
        tasks=GENERATIVE_TASKS + ("anomaly_detection",), conditional_capable=False,
        hp_ranges={"n_blocks": (2, 16), "hidden_dim": (16, 512), **COMMON_TRAINING},
        defaults={"n_blocks": 5, "hidden_dim": 128, "lr": 1e-3, "batch_size": 256,
                  "max_epochs": 300, "weight_decay": 1e-5},
        representations=("standardized",),
        module="flows", cls="MAF",
        summary="Masked autoregressive flow; exact log-likelihood, slow sampling",
        has_likelihood=True, notes=("ambient dimension must be modest (<= 200)",),
    ),
    ModelInfo(
        name="ddpm_unet", family="diffusion", modalities=("image",),
        tasks=GENERATIVE_TASKS, conditional_capable=True,
        hp_ranges={"base_channels": (16, 128), "timesteps": (100, 2000),
                   "sample_steps": (10, 1000), "guidance_scale": (0.0, 10.0),
                   "p_uncond": (0.0, 0.5), **COMMON_TRAINING},
        defaults={"base_channels": 64, "timesteps": 1000, "sample_steps": 100,
                  "guidance_scale": 2.0, "p_uncond": 0.1, "lr": 2e-4, "batch_size": 128,
                  "max_epochs": 200, "weight_decay": 0.0},
        representations=("pixels",),
        module="diffusion", cls="ImageDDPM",
        summary="DDPM with a small U-Net; classifier-free guidance for class conditioning",
    ),
    ModelInfo(
        name="ddpm_mlp", family="diffusion", modalities=("tabular", "counts"),
        tasks=GENERATIVE_TASKS, conditional_capable=True,
        hp_ranges={"hidden_dim": (32, 2048), "n_layers": (2, 8), "timesteps": (50, 2000),
                   "sample_steps": (10, 1000), "guidance_scale": (0.0, 10.0),
                   "p_uncond": (0.0, 0.5), **COMMON_TRAINING},
        defaults={"hidden_dim": 256, "n_layers": 4, "timesteps": 1000, "sample_steps": 100,
                  "guidance_scale": 1.0, "p_uncond": 0.1, "lr": 1e-3, "batch_size": 256,
                  "max_epochs": 300, "weight_decay": 0.0},
        representations=("standardized", "lognorm_pca"),
        module="diffusion", cls="VectorDDPM",
        summary="DDPM with an MLP denoiser for tabular vectors or a PCA/latent space",
    ),
]}

# Maximum ambient dimension for models that work directly in feature space with
# full-rank transforms; beyond this the planner must reduce dimension first.
MAX_FLOW_DIM = 200
MAX_MODELS_PER_PLAN = 3


def allowed(modality: str, task: str) -> list[str]:
    return [m.name for m in REGISTRY.values() if modality in m.modalities and task in m.tasks]


def clip_hyperparameters(name: str, hps: dict) -> tuple[dict, list[str]]:
    """Clip to registry ranges and drop unknown keys. Returns (clean, notes)."""
    info = REGISTRY[name]
    ranges = info.hp_ranges
    clean, notes = {}, []
    for key, value in {**info.defaults, **hps}.items():
        if key not in ranges:
            if key in hps and key not in info.defaults:
                notes.append(f"{name}: dropped unknown hyperparameter '{key}'")
                continue
            clean[key] = value
            continue
        lo, hi = ranges[key]
        try:
            v = float(value)
        except (TypeError, ValueError):
            notes.append(f"{name}: non-numeric {key}={value!r} replaced by default")
            v = float(info.defaults.get(key, lo))
        clipped = min(max(v, lo), hi)
        if clipped != v:
            notes.append(f"{name}: {key}={value} outside [{lo}, {hi}], clipped to {clipped:g}")
        is_int = isinstance(info.defaults.get(key), int) and not isinstance(
            info.defaults.get(key), bool)
        clean[key] = int(round(clipped)) if is_int else clipped
    return clean, notes


def build(spec, **kwargs):
    """Instantiate the implementation of ``spec.name`` (lazy torch import)."""
    import importlib

    info = REGISTRY[spec.name]
    module = importlib.import_module(f"genmodel_agent.models.{info.module}")
    return getattr(module, info.cls)(spec=spec, **kwargs)
