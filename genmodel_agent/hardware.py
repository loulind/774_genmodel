"""Device detection and seeding.

The agent never assumes a GPU. It detects what is present, records it in the run
manifest, and the config module scales every training budget by a per-device
factor -- so the same plan is affordable on a laptop CPU and simply trains
longer on CUDA.
"""

from __future__ import annotations

import os
import platform
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class HardwareInfo:
    device: str            # "cuda" | "mps" | "cpu"
    device_name: str
    cpu: str
    n_cpu: int
    memory_gb: float | None
    torch_version: str

    def to_record(self) -> dict:
        return asdict(self)


def _memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def detect(prefer: str | None = None) -> HardwareInfo:
    """Return the best available device, or ``prefer`` if given and available.

    Order is CUDA > MPS > CPU. ``prefer="cpu"`` is honoured unconditionally,
    which is how tests and fully deterministic reruns opt out of the GPU.
    """
    if prefer == "cpu":
        device = "cpu"
    elif prefer in (None, "auto", "cuda") and torch.cuda.is_available():
        device = "cuda"
    elif prefer in (None, "auto", "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    elif device == "mps":
        name = f"Apple GPU (MPS) on {platform.machine()}"
    else:
        name = "CPU"

    return HardwareInfo(
        device=device,
        device_name=name,
        cpu=platform.processor() or platform.machine(),
        n_cpu=os.cpu_count() or 1,
        memory_gb=_memory_gb(),
        torch_version=torch.__version__,
    )


def seed_everything(seed: int) -> None:
    """Fix every RNG the pipeline touches.

    GPU kernels (MPS especially) are not bit-for-bit deterministic, so reruns
    on the same device agree to within floating-point noise rather than exactly;
    decisions, splits and subsamples are exactly reproducible because they are
    drawn from these seeds on the CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
