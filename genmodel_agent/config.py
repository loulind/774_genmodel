"""Run configuration: budgets, seeds and LLM settings -- never modeling decisions.

A config file may say *how much* compute a run gets, *which* LLM to ask, and
*what* seed to use. It may not say which model to fit; that comes only from the
profile and the objective. Keeping the two apart is what stops a per-dataset
config from becoming a dataset-specific branch in disguise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Multiplier on every time budget, relative to the reference machine (an M2 Pro
# on MPS). A CPU-only run gets less wall-clock per model so a full run still
# finishes in about an hour; it will simply train for fewer epochs.
DEVICE_BUDGET_SCALE = {"mps": 1.0, "cuda": 1.0, "cpu": 0.5}


class Budget(BaseModel):
    """Compute caps for one mode (full or quick)."""
    train_seconds_image: float = 1500      # per model, image modality
    train_seconds_other: float = 480       # per model, tabular / counts
    max_epochs: int = 400
    max_train_samples: int | None = None   # subsample cap before training
    n_synthetic_max: int = 10_000          # samples generated for evaluation
    eval_max_samples: int = 5_000          # rows used by O(n^2) metrics
    device: str = "mps"                    # set by RunConfig.budget(); planners read it


class LLMConfig(BaseModel):
    enabled: bool = True
    backend: str = "claude_agent_sdk"
    # Checked against the id the backend reports back; see llm.py.
    model: str = "claude-opus-5-5"
    max_output_tokens: int = 8_000
    max_retries: int = 2


class RunConfig(BaseModel):
    seed: int = 774
    device: str = "auto"                    # auto | cuda | mps | cpu
    quick: bool = False
    replay: bool = False                    # reuse cached LLM responses only
    runs_dir: str = "runs"
    label_column: str | None = None
    max_critic_revisions: int = 1
    full: Budget = Field(default_factory=Budget)
    quick_budget: Budget = Field(default_factory=lambda: Budget(
        train_seconds_image=60, train_seconds_other=45, max_epochs=3,
        max_train_samples=2_000, n_synthetic_max=1_000, eval_max_samples=1_000,
    ))
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def budget(self, device: str) -> Budget:
        """The active budget, scaled for the detected device."""
        base = self.quick_budget if self.quick else self.full
        scale = DEVICE_BUDGET_SCALE.get(device, 1.0)
        return base.model_copy(update={
            "device": device,
            "train_seconds_image": base.train_seconds_image * scale,
            "train_seconds_other": base.train_seconds_other * scale,
        })


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> RunConfig:
    """Load default.yaml, layer an optional file on top, then keyword overrides."""
    data: dict = {}
    if DEFAULT_CONFIG.exists():
        data = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if path is not None:
        data = _deep_update(data, yaml.safe_load(Path(path).read_text()) or {})
    data = _deep_update(data, {k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**data)
