"""End-to-end: run_agent on small synthetic files (CPU, rule-based, seconds per model)."""

import json

import numpy as np
import pandas as pd
import pytest

from genmodel_agent import run_agent

TINY = {"train_seconds_image": 3, "train_seconds_other": 3, "max_epochs": 30,
        "max_train_samples": 400, "n_synthetic_max": 300, "eval_max_samples": 300}
EXPECTED = ["report.md", "report.html", "decision_log.json", "plan.json", "profile.json",
            "evaluation.json", "model_summary.json", "recommendations.json", "run_manifest.json"]


def _table(tmp_path, n=360, rare=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < rare, "rare", "common")
    df = pd.DataFrame({"a": rng.normal(size=n) + (y == "rare") * 2,
                       "b": rng.gamma(2, 1, n), "c": rng.normal(size=n),
                       "grp": rng.choice(["x", "y", "z"], n), "label": y})
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("objective, task", [
    ("Generate synthetic rows for the rare class to rebalance a downstream classifier",
     "class_balancing"),
    ("Detect anomalous rows", "anomaly_detection"),
])
def test_run_agent_tabular(tmp_path, objective, task):
    out = tmp_path / "out"
    rare = 0.08 if task == "class_balancing" else 0.03
    res = run_agent(_table(tmp_path, rare=rare), objective, out, quick=True, no_llm=True,
                    device="cpu", quick_budget=TINY, runs_dir=str(tmp_path / "runs"))
    assert res["plan"]["task"]["task_type"] == task and res["plan"]["use_generative_model"]
    for f in EXPECTED:
        assert (out / f).exists(), f
    log = json.loads((out / "decision_log.json").read_text())
    steps = {e["step"] for e in log}
    assert {"profile", "objective", "warrant", "model", "hyperparameters", "critic"} <= steps
    assert all(e["rationale"] for e in log if e["step"] in ("warrant", "model"))
    assert any((out / "synthetic_sample").iterdir())
    assert "## 9. Reproducibility" in (out / "report.md").read_text()


def test_run_agent_declines(tmp_path):
    out = tmp_path / "out"
    res = run_agent(_table(tmp_path, rare=0.4), "Generate synthetic records to augment the data "
                    "and improve a classifier", out, quick=True, no_llm=True, device="cpu",
                    quick_budget=TINY, runs_dir=str(tmp_path / "runs"))
    assert not res["plan"]["use_generative_model"]
    assert res["results"]["alternative"]["macro_f1_mean"] > 0.5
    assert "no generative model" in (out / "report.md").read_text()
