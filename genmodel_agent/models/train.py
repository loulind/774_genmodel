"""Shared training loop: wall-clock and epoch caps, early stopping, history.

Every model trains through ``train_loop``. It owns the stopping rules, so they
are identical across model families and every run reports *why* training
stopped: "time" (budget hit), "epochs" (epoch cap), or "early_stop" (validation
loss stopped improving for ``patience`` epochs). The best-validation weights
are restored at the end, which is the usual guard against overfitting late in
training -- an especially real risk at the small n of rare classes.

Models supply two callables:
    step(batch_index_tensor) -> dict of floats   one optimisation step
    validate() -> float | None                    validation loss (None: no early stopping)
"""

from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import torch


def batches(n: int, batch_size: int, generator: torch.Generator, device) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    bs = max(1, min(batch_size, n))
    # Drop a last batch smaller than half a batch: tiny batches give noisy
    # gradients and break batch statistics.
    chunks = [perm[i:i + bs] for i in range(0, n, bs)]
    if len(chunks) > 1 and len(chunks[-1]) < bs // 2:
        chunks = chunks[:-1]
    return [c.to(device) for c in chunks]


def train_loop(step: Callable[[torch.Tensor], dict], n_train: int, batch_size: int,
               max_epochs: int, max_seconds: float, modules: list[torch.nn.Module],
               validate: Callable[[], float | None] | None = None,
               patience: int = 25, seed: int = 0, device="cpu",
               log_every: int = 0, on_epoch_end: Callable[[int], None] | None = None) -> dict:
    gen = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    curve, best, best_epoch, best_state = [], np.inf, -1, None
    reason, epoch = "epochs", 0
    for epoch in range(1, int(max_epochs) + 1):
        for m in modules:
            m.train()
        logs: dict[str, list[float]] = {}
        for idx in batches(n_train, batch_size, gen, device):
            for k, v in step(idx).items():
                logs.setdefault(k, []).append(float(v))
            if time.perf_counter() - start > max_seconds:
                break
        record = {"epoch": epoch, **{k: float(np.mean(v)) for k, v in logs.items()}}
        if validate is not None:
            for m in modules:
                m.eval()
            with torch.no_grad():
                val = validate()
            if val is not None:
                record["val_loss"] = float(val)
                if val < best - 1e-4 * abs(best if np.isfinite(best) else 1):
                    best, best_epoch = val, epoch
                    best_state = [copy.deepcopy(m.state_dict()) for m in modules]
        if on_epoch_end:
            on_epoch_end(epoch)
        curve.append(record)
        if log_every and epoch % log_every == 0:
            print(f"      epoch {epoch}: " + ", ".join(f"{k}={v:.4g}" for k, v in record.items()
                                                       if k != "epoch"), flush=True)
        if time.perf_counter() - start > max_seconds:
            reason = "time"
            break
        if best_epoch > 0 and epoch - best_epoch >= patience:
            reason = "early_stop"
            break
    if best_state is not None:
        for m, s in zip(modules, best_state):
            m.load_state_dict(s)
    for m in modules:
        m.eval()
    return {
        "epochs_run": epoch, "seconds": round(time.perf_counter() - start, 1),
        "stopped_because": reason, "best_epoch": best_epoch if best_state else epoch,
        "best_val_loss": None if best_state is None else float(best),
        "final_train_loss": next((v for k, v in curve[-1].items() if k.endswith("loss")
                                  and k != "val_loss"), None) if curve else None,
        "curve": curve,
    }
