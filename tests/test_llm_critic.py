"""LLM plumbing (with a fake backend), LLM planning fallbacks, and critic triggers."""

import json

import pytest

from genmodel_agent.critic import _rule_critique, find_triggers, primary_metric_value
from genmodel_agent.decision_log import DecisionLog
from genmodel_agent.llm import LLMClient, LLMResponse, LLMUnavailable, extract_json
from genmodel_agent.objective import parse_objective
from genmodel_agent.planner import heuristic
from genmodel_agent.planner import llm as llm_planner
from genmodel_agent.schema import EvalPlan, ModelSpec, Plan, TaskSpec
from test_planner import BUDGET, make_profile


class FakeBackend:
    name, model = "fake", "fake-model"

    def __init__(self, replies):
        self.replies, self.n = list(replies), 0

    def complete(self, system, user):
        self.n += 1
        return LLMResponse(text=self.replies.pop(0), backend="fake", model_requested="fake-model",
                           model_used="fake-model", seconds=0.0)


def test_extract_json_fenced_and_bare():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise {"b": [1, 2]} trailing') == {"b": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_cache_and_replay(tmp_path):
    backend = FakeBackend(['{"x": 1}'])
    c = LLMClient(backend, tmp_path / "cache")
    assert c.complete("r", "s", "u").text == '{"x": 1}'
    assert c.complete("r", "s", "u").cached and backend.n == 1          # second call: cache
    replay = LLMClient(FakeBackend([]), tmp_path / "new", replay_from=tmp_path / "cache")
    assert replay.complete("r", "s", "u").text == '{"x": 1}'
    # Prompt drift (e.g. different incidental digits) falls back to the role's response...
    assert replay.complete("r", "s", "different prompt").text == '{"x": 1}'
    # ...but a role that was never called cannot be replayed.
    with pytest.raises(LLMUnavailable):
        replay.complete("other_role", "s", "u")


def _setup():
    prof = make_profile(n=5000, p=8, classes={"n": 4900, "a": 100})
    task = parse_objective("Detect anomalous cases", prof)
    proposal = heuristic.plan(prof, task, BUDGET, DecisionLog())
    return prof, task, proposal


def test_llm_plan_modification_logged(tmp_path):
    prof, task, proposal = _setup()
    changed = proposal.model_dump()
    changed["models"] = [m for m in changed["models"] if m["role"] == "primary"]
    changed["models"][0]["hyperparameters"]["n_blocks"] = 8
    reply = json.dumps({"plan": changed, "decisions": [
        {"step": "model", "decision": "drop comparison", "chosen": "maf only",
         "rationale": "budget", "alternatives_rejected": [{"option": "vae", "reason": "redundant"}]}]})
    log = DecisionLog()
    plan = llm_planner.plan(prof, task, BUDGET, log, LLMClient(FakeBackend([reply]), tmp_path))
    assert plan.source == "llm" and len(plan.models) == 1
    assert plan.models[0].hyperparameters["n_blocks"] == 8
    assert any(e.source == "llm" and "n_blocks: 5 -> 8" in e.rationale for e in log.entries)


def test_llm_plan_repaired_then_fallback(tmp_path):
    prof, task, proposal = _setup()
    bad = proposal.model_dump()
    bad["models"][0]["name"] = "transformer"
    good = proposal.model_dump()
    plan = llm_planner.plan(prof, task, BUDGET, DecisionLog(),
                            LLMClient(FakeBackend([json.dumps({"plan": bad}),
                                                   json.dumps({"plan": good})]), tmp_path / "a"))
    assert plan.source == "llm_repaired"
    log = DecisionLog()
    plan = llm_planner.plan(prof, task, BUDGET, log,
                            LLMClient(FakeBackend(["not json", "still not json"]), tmp_path / "b"))
    assert plan.source == "heuristic_fallback" and plan.models[0].name == "maf"
    assert any("fallback" in e.chosen for e in log.entries)


def _gen_plan(primary="ddpm_unet", other="vae", metric="macro_f1_gain"):
    return Plan(use_generative_model=True, decision_rationale="x",
                task=TaskSpec(raw_objective="x", task_type="augmentation"),
                models=[ModelSpec(name=primary, role="primary",
                                  hyperparameters={"guidance_scale": 2.0}),
                        ModelSpec(name=other, role="comparison")],
                evaluation=EvalPlan(metrics=["precision_recall"], primary_metric=metric))


def _gen_results(mem=False, recall=0.6, gain_p=0.02, gain_o=0.01):
    gain = lambda g: {"macro_f1_gain": {"gain": g, "ci_low": g - .02, "ci_high": g + .02}}
    return {"generative": {
        "reference_real_holdout": {"recall": 0.6},
        "models": {"ddpm_unet": {"recall": recall, "memorization_flag": mem, "too_close_fraction": .3},
                   "vae": {"recall": 0.5, "memorization_flag": False}},
        "utility": {"models": {"ddpm_unet": gain(gain_p), "vae": gain(gain_o)}}}}


def test_critic_triggers_and_priority():
    plan = _gen_plan()
    assert _rule_critique(find_triggers(plan, _gen_results())).action == "none"
    c = _rule_critique(find_triggers(plan, _gen_results(mem=True, recall=0.1, gain_o=0.05)))
    assert c.action == "switch" and c.target_model == "vae"          # memorization first
    c = _rule_critique(find_triggers(plan, _gen_results(recall=0.1)))
    assert c.action == "retune" and c.hyperparameter_changes == {"guidance_scale": 1.0}
    assert primary_metric_value(_gen_results(), plan, "vae") == pytest.approx(0.01)


def test_critic_baseline_better_for_imputation():
    plan = Plan(use_generative_model=True, decision_rationale="x",
                task=TaskSpec(raw_objective="x", task_type="imputation"),
                models=[ModelSpec(name="masked_vae", role="primary",
                                  hyperparameters={"latent_dim": 8, "weight_decay": 1e-4})],
                evaluation=EvalPlan(metrics=["masked_rmse"], primary_metric="masked_rmse"))
    res = {"imputation": {"methods": {"masked_vae": {"masked_rmse": 0.9},
                                      "mice": {"masked_rmse": 0.7}, "mean_mode": {"masked_rmse": 1.0}}}}
    c = _rule_critique(find_triggers(plan, res))
    assert c.action == "retune" and c.hyperparameter_changes["latent_dim"] == 4
