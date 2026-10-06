"""The agent loop, end to end.

    load -> profile -> parse objective -> plan (LLM | rules) -> validate
         -> split -> [per model] represent, train, sample -> evaluate
         -> critic -> at most one revision (kept only if it helps)
         -> recommendations -> report

Every decision goes to ``decision_log.json`` as it is made. Heavy scratch
(LLM cache, logs) lives in ``runs/<timestamp>_<name>/``; the compact deliverables
go to ``out_dir``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import critic as critic_mod
from . import figures, hardware, recommend, report
from .config import REPO_ROOT, load_config
from .data.loaders import load_path
from .data.splits import Subset, make_splits
from .decision_log import DecisionLog
from .evaluate import evaluate_generative
from .evaluate.features import EvalSpace
from .evaluate.task import anomaly_eval, imputation_eval, representation_eval
from .llm import make_client
from .models.registry import REGISTRY, build, clip_hyperparameters
from .objective import parse_objective
from .planner import llm as llm_planner
from .preprocess import build_representation
from .profiler import profile as profile_data
from .schema import ModelSpec, Plan

GENERATIVE = ("generation", "augmentation", "class_balancing")
SAMPLE_CAP_IMAGES_PER_CLASS = 40
SAMPLE_CAP_ROWS = 500
SAMPLE_CAP_CELLS = 300


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


def _json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    return str(o)


class Agent:
    def __init__(self, data_path, objective, out_dir, **overrides):
        self.t0 = time.perf_counter()
        config_path = overrides.pop("config", None)
        no_llm = overrides.pop("no_llm", False)
        self.cfg = load_config(config_path, **overrides)
        if no_llm:
            self.cfg.llm.enabled = False
        self.data_path, self.objective = Path(data_path), objective
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = REPO_ROOT / self.cfg.runs_dir / f"{stamp}_{self.data_path.stem}"
        self.say = RunLog(self.run_dir / "run.log")
        self.log = DecisionLog(self.out / "decision_log.json")
        self.hw = hardware.detect(self.cfg.device)
        hardware.seed_everything(self.cfg.seed)
        self.budget = self.cfg.budget(self.hw.device)
        self.seeds = [self.cfg.seed + i for i in range(1 if self.cfg.quick else 3)]
        self.classifier_seconds = (10 if self.cfg.quick else 60) * \
            (self.budget.train_seconds_other / (self.cfg.quick_budget if self.cfg.quick
                                                else self.cfg.full).train_seconds_other)
        self.client = make_client(self.cfg.llm.enabled, self.cfg.llm.model,
                                  self.run_dir / "llm_cache", self.cfg.replay,
                                  self.cfg.llm.max_output_tokens)
        self.models, self.reps, self.histories = {}, {}, {}
        self.synth: dict = {}
        self.results: dict = {}
        self.revision: dict = {"ran": False}
        self.figs: dict[str, str] = {}

    # ------------------------------------------------------------------ steps
    def run(self) -> dict:
        say = self.say
        say(f"device {self.hw.device} ({self.hw.device_name}); mode {'quick' if self.cfg.quick else 'full'}; "
            f"LLM {'off' if not self.client.available else self.client.model}")
        self.log.add("system", "compute", f"{self.hw.device}, {'quick' if self.cfg.quick else 'full'} mode",
                     f"Detected {self.hw.device_name}; per-model training caps "
                     f"{self.budget.train_seconds_image:.0f}s (images) / "
                     f"{self.budget.train_seconds_other:.0f}s (other); seed {self.cfg.seed}.",
                     evidence=self.hw.to_record(), source="system")

        self.data = load_path(self.data_path, self.cfg.label_column)
        self.profile = profile_data(self.data, self.cfg.seed)
        _json(self.profile.model_dump(), self.out / "profile.json")
        say(f"profile: {self.profile.brief()}")
        self.log.add("profile", "data modality", self.profile.modality,
                     self.profile.modality_evidence,
                     evidence={"n": self.profile.n_samples, "p": self.profile.n_features,
                               "brief": self.profile.brief()}, source="system")

        task = parse_objective(self.objective, self.profile)
        self.log.add("objective", "task type", task.task_type,
                     f"Matched phrases: {task.evidence or 'none (default: generation)'}.",
                     [(t, "lower precedence") for t in task.secondary_tasks],
                     {"conditional": task.conditional, "label_budget": task.label_budget,
                      "target_classes": task.target_classes, "wants_tstr": task.wants_tstr})

        self.plan = llm_planner.plan(self.profile, task, self.budget, self.log, self.client)
        self._apply_budget_caps()
        _json(self.plan.model_dump(), self.out / "plan.json")
        say(f"plan ({self.plan.source}): generate={self.plan.use_generative_model}; "
            f"models={[(m.name, m.role) for m in self.plan.models]}")

        if not self.plan.use_generative_model:
            self.results["alternative"] = self._alternative_analysis()
        else:
            self.splits = make_splits(self.data, self.plan.data, self.cfg.seed)
            for note in self.splits.notes:
                say(f"split: {note}")
            self.log.add("preprocessing", "data split",
                         f"train {len(self.splits.train)} / val {len(self.splits.val)} / "
                         f"test {len(self.splits.test)}", "; ".join(self.splits.notes),
                         source="system")
            for spec in self.plan.models:
                self._train(spec)
            self.results = self._evaluate(self.plan.models)
            self._critic_and_revise()

        self.recs = recommend.recommend(self.plan, self.results, self.profile)
        self.log.add("recommendation", "verdict", self.recs.get("verdict", "")[:120],
                     self.recs.get("verdict", ""), source="system")
        self._figures()
        return self._write_outputs()

    def _apply_budget_caps(self) -> None:
        for m in self.plan.models:
            hp = m.hyperparameters
            if "max_epochs" in hp and hp["max_epochs"] > self.budget.max_epochs:
                self.log.add("validator", f"{m.name} epoch cap",
                             f"max_epochs {hp['max_epochs']} -> {self.budget.max_epochs}",
                             "Run-level epoch cap for this mode; wall-clock cap "
                             f"{m.train_seconds:.0f}s still applies.", source="validator")
                hp["max_epochs"] = int(self.budget.max_epochs)
        if self.budget.max_train_samples:
            self.plan.data.max_train_samples = self.budget.max_train_samples

    # ------------------------------------------------------------ training
    def _class_index(self, name: str) -> int:
        return self.data.class_names.index(name)

    def _train_subset(self, subset: Subset) -> Subset:
        """Anomaly detection trains on the normal class(es) only."""
        if self.plan.task.task_type == "anomaly_detection" and subset.y is not None:
            anomaly = self._class_index(self.profile.labels.minority_class)
            keep = subset.y != anomaly
            X = subset.X.iloc[keep].reset_index(drop=True) if isinstance(subset.X, pd.DataFrame) \
                else subset.X[keep]
            return Subset(X, subset.y[keep], subset.index[keep])
        return subset

    def _train(self, spec: ModelSpec, tag: str | None = None) -> None:
        name = tag or spec.name
        tr, va = self._train_subset(self.splits.train), self._train_subset(self.splits.val)
        kw = {"max_modes": spec.hyperparameters.get("max_modes", 5), "seed": self.cfg.seed} \
            if spec.representation == "mode_specific" else {}
        rep = build_representation(spec.representation, n_hvg=self.plan.data.n_hvg, **kw).fit(tr)
        Xtr, Xv = rep.transform(tr.X), rep.transform(va.X)
        ex_tr, ex_v = rep.extras(tr.X), rep.extras(va.X)
        if spec.name == "masked_vae":
            ex_tr["mask"], ex_v["mask"] = rep.observed_mask(tr.X), rep.observed_mask(va.X)
        n_classes = self.profile.labels.n_classes if self.profile.labels else 0
        model = build(spec, info=rep.info(), n_classes=n_classes, device=self.hw.device,
                      seed=self.cfg.seed)
        self.say(f"train {name}: {spec.representation}, n={len(Xtr)}, "
                 f"cap {spec.train_seconds:.0f}s, {model.n_parameters():,} params")
        hist = model.fit(Xtr, tr.y if spec.conditional else None, Xv,
                         va.y if spec.conditional else None, ex_tr, ex_v)
        self.say(f"  {name}: {hist['epochs_run']} epochs, {hist['seconds']}s, "
                 f"stopped: {hist['stopped_because']}")
        self.models[name], self.reps[name], self.histories[name] = model, rep, hist
        if self.plan.task.task_type in GENERATIVE:
            self._sample(name, spec)

    def _sample(self, name: str, spec: ModelSpec) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        model, rep = self.models[name], self.reps[name]
        y_train = self.splits.train.y
        n_eval = int(min(len(self.splits.train), self.budget.n_synthetic_max))
        y_eval = None
        if spec.conditional and y_train is not None:
            y_eval = rng.choice(y_train, n_eval, replace=len(y_train) < n_eval)
        t = time.perf_counter()
        Z = model.sample(n_eval, y_eval)
        augment = None
        if spec.conditional and self.plan.sampling.n_per_class:
            y_aug = np.concatenate([np.full(k, self._class_index(c))
                                    for c, k in self.plan.sampling.n_per_class.items() if k > 0]
                                   or [np.zeros(0, int)]).astype(int)
            augment = (model.sample(len(y_aug), y_aug), y_aug) if len(y_aug) else None
        self.say(f"  {name}: sampled {n_eval} (+{0 if augment is None else len(augment[1])} "
                 f"augmentation) in {time.perf_counter() - t:.0f}s")
        self.synth[name] = {"Z": Z, "y": y_eval, "augment": augment}

    # ------------------------------------------------------------ evaluation
    def _evaluate(self, specs: list[ModelSpec]) -> dict:
        t = self.plan.task.task_type
        names = [s.name for s in specs]
        if t in GENERATIVE:
            return {"generative": self._evaluate_generative(names)}
        if t == "imputation":
            out = {"methods": {}}
            for n in names:
                r = imputation_eval(self.reps[n], self.models[n], self.splits.train.X,
                                    self.splits.test.X, self.cfg.seed)
                out["methods"].update(r["methods"])
                out.update({k: v for k, v in r.items() if k != "methods"})
            return {"imputation": out}
        if t == "anomaly_detection":
            anomaly = self._class_index(self.profile.labels.minority_class)
            y_test = (self.splits.test.y == anomaly).astype(int)
            scores, X_ref = {}, None
            for n in names:
                rep = self.reps[n]
                scores[n] = -self.models[n].score(rep.transform(self.splits.test.X))
                X_ref = X_ref if X_ref is not None else rep
            normal = self._train_subset(self.splits.train)
            res = anomaly_eval(scores, X_ref.transform(normal.X), X_ref.transform(self.splits.test.X),
                               y_test, self.cfg.seed)
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(random_state=self.cfg.seed).fit(X_ref.transform(normal.X))
            self.anomaly_scores = {**scores, "isolation_forest":
                                   -iso.score_samples(X_ref.transform(self.splits.test.X))}
            self.anomaly_y = y_test
            res["anomaly_class"] = self.profile.labels.minority_class
            return {"anomaly": res}
        if t == "representation_learning":
            out = {}
            for n in names:
                rep = self.reps[n]
                Xtr, Xte = rep.transform(self.splits.train.X), rep.transform(self.splits.test.X)
                out[n] = representation_eval(self.models[n].encode(Xtr), self.models[n].encode(Xte),
                                             Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1),
                                             self.splits.train.y, self.splits.test.y, self.cfg.seed)
            return {"representation": out}
        return {}

    def _eval_space(self) -> EvalSpace:
        if not hasattr(self, "space"):
            self.say("fitting evaluation feature space")
            self.space = EvalSpace(self.profile.modality, self.hw.device, self.cfg.seed,
                                   self.plan.data.n_hvg, self.classifier_seconds) \
                .fit(self.splits.train, self.splits.val)
            self.log.add("evaluation", "feature space", self.space.description,
                         "Fitted on real training data only.", source="system")
            sp = self.splits
            from .evaluate.utility import concat
            predefined = bool(self.data.splits) and self.plan.data.use_predefined_splits
            val_c = self.space.real_canonical(sp.val.X)
            test_c = self.space.real_canonical(sp.test.X)
            hold = val_c if predefined else concat(val_c, test_c)
            y_hold = sp.val.y if predefined or sp.val.y is None else np.concatenate([sp.val.y, sp.test.y])
            self.real = {"holdout": hold, "y_holdout": y_hold, "test_is_separate": predefined,
                         "holdout_definition": "validation split (IID with train; predefined test "
                         "split reported separately)" if predefined else
                         "validation + test splits (both drawn IID by the agent)",
                         "train": self.space.real_canonical(sp.train.X), "y_train": sp.train.y,
                         "val": val_c, "y_val": sp.val.y, "test": test_c, "y_test": sp.test.y,
                         "train_counts": self.space.real_counts(sp.train.X)}
        return self.space

    def _evaluate_generative(self, names: list[str]) -> dict:
        space = self._eval_space()
        synth = {}
        for n in names:
            s, rep = self.synth[n], self.reps[n]
            aug = None if s["augment"] is None else \
                (space.model_canonical(rep, s["augment"][0]), s["augment"][1])
            synth[n] = {"eval": (space.model_canonical(rep, s["Z"]), s["y"]), "augment": aug,
                        "counts": s["Z"] if rep.name == "counts_hvg" else None}
        self.synth_canonical = {**getattr(self, "synth_canonical", {}), **synth}
        targets = [self._class_index(c) for c in self.plan.sampling.target_classes] \
            if self.plan.task.task_type == "class_balancing" else []
        n_classes = self.profile.labels.n_classes if self.profile.labels else 0
        self.say(f"evaluating {names} ({len(self.seeds)} classifier seed(s))")
        t = self.plan.task
        return evaluate_generative(
            space, self.real, synth, self.plan.evaluation.metrics, targets, self.data.class_names,
            n_classes, self.seeds, t.wants_tstr or t.task_type in ("generation", "augmentation"),
            self.budget.eval_max_samples, self.cfg.seed, self.plan.evaluation.baselines)

    def _alternative_analysis(self) -> dict:
        """When generation is declined, run the recommended baseline so the report has numbers."""
        if self.data.y is None or self.profile.modality == "image":
            return {"summary": "no labels to evaluate a baseline on"}
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        from .preprocess import Tabular

        X, y = self.data.X, self.data.y
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X.todense() if hasattr(X, "todense") else X))
        f1s, bals, aucs = [], [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=self.cfg.seed).split(X, y):
            enc = Tabular(keep_missing=False).fit(Subset(X.iloc[tr].reset_index(drop=True), y[tr], tr))
            clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
            clf.fit(enc.transform(X.iloc[tr]), y[tr])
            pred = clf.predict(enc.transform(X.iloc[te]))
            f1s.append(f1_score(y[te], pred, average="macro"))
            bals.append(balanced_accuracy_score(y[te], pred))
            if len(np.unique(y)) == 2:
                aucs.append(roc_auc_score(y[te], clf.predict_proba(enc.transform(X.iloc[te]))[:, 1]))
        out = {"method": "class-weighted logistic regression, standardised features, 5-fold "
                         "stratified CV (encoder fitted within folds)",
               "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
               "balanced_accuracy_mean": float(np.mean(bals)),
               "auroc_mean": float(np.mean(aucs)) if aucs else None,
               "auroc_sd": float(np.std(aucs)) if aucs else None}
        out["summary"] = (f"{out['method']}: macro-F1 {out['macro_f1_mean']:.3f} "
                          f"(sd {out['macro_f1_sd']:.3f})"
                          + (f", AUROC {out['auroc_mean']:.3f} (sd {out['auroc_sd']:.3f})"
                             if aucs else ""))
        self.say(f"alternative analysis: {out['summary']}")
        self.log.add("alternative", "baseline analysis run", out["method"], out["summary"],
                     source="system")
        return out

    # ------------------------------------------------------------ critic
    def _critic_and_revise(self) -> None:
        training = {n: {k: v for k, v in h.items() if k not in ("curve", "val_curve")}
                    for n, h in self.histories.items()}
        cap = self.budget.train_seconds_image if self.profile.modality == "image" \
            else self.budget.train_seconds_other
        self.critique = critic_mod.critique(
            self.plan, self.profile, self.results, training, self.client, self.log,
            {"max_epochs_cap": self.budget.max_epochs, "per_model_train_seconds": cap,
             "note": "a revision retrains one model within these caps"})
        self.say(f"critic ({self.critique.source}): {self.critique.action} "
                 f"{self.critique.target_model or ''} {self.critique.hyperparameter_changes or ''}")
        c = self.critique
        if c.action == "none" or self.cfg.max_critic_revisions < 1 or not c.target_model:
            return
        primary = next(m for m in self.plan.models if m.role == "primary")
        before = critic_mod.primary_metric_value(self.results, self.plan, c.target_model) \
            if c.target_model in self.models else None
        before_primary = critic_mod.primary_metric_value(self.results, self.plan, primary.name)
        existing = next((m for m in self.plan.models if m.name == c.target_model), None)
        if c.action == "retune" or existing is None or c.hyperparameter_changes:
            base = existing or ModelSpec(name=c.target_model, role="comparison",
                                         conditional=primary.conditional and
                                         REGISTRY[c.target_model].conditional_capable,
                                         representation=REGISTRY[c.target_model].representations[0],
                                         train_seconds=primary.train_seconds)
            hp, _ = clip_hyperparameters(c.target_model, {**base.hyperparameters,
                                                           **c.hyperparameter_changes})
            hp["max_epochs"] = int(min(hp.get("max_epochs", 2000), self.budget.max_epochs))
            new_spec = base.model_copy(update={"hyperparameters": hp})
            old_model = (self.models.get(c.target_model), self.reps.get(c.target_model),
                         self.histories.get(c.target_model), self.synth.get(c.target_model))
            old_canonical = getattr(self, "synth_canonical", {}).get(c.target_model)
            self._train(new_spec)
            new_results = self._evaluate([new_spec])
            trial = _merge(self.results, new_results)
            after = critic_mod.primary_metric_value(trial, self.plan, new_spec.name)
            # A switch must beat the current primary; a retune must beat its own past self.
            reference = before_primary if c.action == "switch" else \
                (before if before is not None else before_primary)
            keep = after is not None and (reference is None or after > reference)
            if keep:
                self.results = trial
                if existing is None:
                    self.plan.models.append(new_spec)
                else:
                    self.plan.models[self.plan.models.index(existing)] = new_spec
                if c.action == "switch":
                    self._promote(new_spec.name)
            else:
                m, r, h, s = old_model
                if m is None:
                    for d in (self.models, self.reps, self.histories, self.synth):
                        d.pop(c.target_model, None)
                else:
                    self.models[c.target_model], self.reps[c.target_model] = m, r
                    self.histories[c.target_model], self.synth[c.target_model] = h, s
                    if old_canonical is not None:
                        self.synth_canonical[c.target_model] = old_canonical
        else:   # switch to an already-trained model
            after = before
            mem_old = self.results.get("generative", {}).get("models", {}).get(primary.name, {}) \
                .get("memorization_flag")
            keep = (after is not None and before_primary is not None and after >= before_primary) \
                or bool(mem_old)
            if keep:
                self._promote(c.target_model)
            reference = before_primary
        self.revision = {"ran": True, "action": c.action, "target": c.target_model,
                         "changes": c.hyperparameter_changes, "metric": self.plan.evaluation.primary_metric,
                         "before": reference, "after": after, "kept": bool(keep)}
        self.say(f"revision {c.action} {c.target_model}: {reference} -> {after}; "
                 f"{'kept' if keep else 'reverted'}")
        self.log.add("revision", f"{c.action} {c.target_model}", "kept" if keep else "reverted",
                     f"Primary metric ({self.plan.evaluation.primary_metric}, higher is better "
                     f"after sign alignment): {reference} -> {after}. A revision is kept only if it "
                     "improves the primary metric.", evidence=self.revision, source="critic")
        _json(self.plan.model_dump(), self.out / "plan.json")

    def _promote(self, name: str) -> None:
        for m in self.plan.models:
            m.role = "primary" if m.name == name else "comparison"
        self.plan.models.sort(key=lambda m: m.role != "primary")

    # ------------------------------------------------------------ figures
    def _figures(self) -> None:
        figdir = self.out / "figures"
        if not self.plan.use_generative_model:
            return
        try:
            self.figs["training"] = figures.training_curves(self.histories, figdir / "training_curves.png")
            t, mod = self.plan.task.task_type, self.profile.modality
            names = [m.name for m in self.plan.models if m.name in self.models]
            if t in GENERATIVE:
                sp = self.splits
                if mod == "image":
                    rep = self.reps[names[0]]
                    real = sp.train.X
                    synth = {}
                    for n in names:
                        imgs = self.reps[n].inverse(self.synth[n]["Z"])
                        synth[n] = (imgs, self.synth[n]["y"] if self.synth[n]["y"] is not None
                                    else np.zeros(len(imgs), int))
                    self.figs["samples"] = figures.image_grid(real, sp.train.y, synth,
                                                              self.data.class_names,
                                                              figdir / "samples_real_vs_synthetic.png")
                F_real = self.space.features(self.real["train"])
                F_syn = {n: self.space.features(self.synth_canonical[n]["eval"][0]) for n in names}
                targets = [self._class_index(c) for c in self.plan.sampling.target_classes] \
                    if t == "class_balancing" else None
                self.figs["embedding"] = figures.embedding_overlay(
                    F_real, F_syn, figdir / "embedding_real_vs_synthetic.png",
                    "Real vs synthetic, PCA of evaluation features",
                    sp.train.y, {n: self.synth_canonical[n]["eval"][1] for n in names}, targets,
                    self.data.class_names)
                if mod == "counts":
                    counts = {n: self.synth_canonical[n]["counts"] for n in names
                              if self.synth_canonical[n]["counts"] is not None}
                    if counts:
                        self.figs["counts"] = figures.count_gene_stats(
                            self.real["train_counts"], counts, figdir / "gene_statistics.png")
                if mod == "tabular":
                    self.figs["marginals"] = figures.table_marginals(
                        self.real["train"], {n: self.synth_canonical[n]["eval"][0] for n in names},
                        figdir / "marginals.png")
                util = self.results["generative"].get("utility")
                if util:
                    self.figs["utility"] = figures.utility_bars(util, figdir / "utility.png",
                                                                self.data.class_names, targets)
            elif t == "imputation":
                self.figs["imputation"] = figures.method_bars(
                    self.results["imputation"]["methods"], "masked_rmse",
                    figdir / "imputation_rmse.png", "Imputation of held-out entries", True, names)
            elif t == "anomaly_detection":
                self.figs["anomaly"] = figures.pr_curves(self.anomaly_y, self.anomaly_scores,
                                                         figdir / "precision_recall.png")
        except Exception as err:   # a figure must never sink a run
            self.say(f"figure generation failed: {err!r}")
            self.log.add("system", "figures", "partial", repr(err), source="system")

    # ------------------------------------------------------------ outputs
    def _write_outputs(self) -> dict:
        out = self.out
        training = {n: self.models[n].summary() for n in self.models}
        _json({"training": training, "revision": self.revision,
               "critique": getattr(self, "critique", None)}, out / "model_summary.json")
        _json(self.results, out / "evaluation.json")
        _json(self.recs, out / "recommendations.json")
        report.write_tables(self.results, out / "tables")
        self._save_samples()
        sections = report.narrative(self)          # before the manifest, so its cost is counted
        manifest = self._manifest()
        _json(manifest, out / "run_manifest.json")
        report.write_report(self, manifest, sections)
        self.say(f"done in {manifest['seconds_total']:.0f}s -> {out}")
        return {"out_dir": str(out), "run_dir": str(self.run_dir), "plan": self.plan.model_dump(),
                "recommendations": self.recs, "revision": self.revision,
                "results": json.loads(json.dumps(self.results, default=_jsonable))}

    def _save_samples(self) -> None:
        if not self.plan.use_generative_model:
            return
        d = self.out / "synthetic_sample"
        d.mkdir(exist_ok=True)
        primary = next(m.name for m in self.plan.models if m.role == "primary")
        t, mod = self.plan.task.task_type, self.profile.modality
        if t in GENERATIVE:
            s, rep = self.synth[primary], self.reps[primary]
            if mod == "image":
                imgs, y = rep.inverse(s["Z"]), s["y"]
                keep = np.concatenate([np.flatnonzero(y == c)[:SAMPLE_CAP_IMAGES_PER_CLASS]
                                       for c in np.unique(y)]) if y is not None \
                    else np.arange(min(len(imgs), 200))
                np.savez_compressed(d / f"{primary}_images.npz", images=imgs[keep],
                                    labels=None if y is None else y[keep],
                                    class_names=np.array(self.data.class_names or []))
            elif mod == "counts":
                genes = np.array(self.data.feature_names)[rep.genes] if hasattr(rep, "genes") \
                    else np.array([])
                idx = np.arange(min(len(s["Z"]), SAMPLE_CAP_CELLS))
                np.savez_compressed(d / f"{primary}_cells.npz", X=s["Z"][idx].astype(np.float32),
                                    labels=None if s["y"] is None else s["y"][idx], genes=genes,
                                    class_names=np.array(self.data.class_names or []),
                                    space=rep.name)
            else:
                df = rep.inverse(s["Z"][:SAMPLE_CAP_ROWS])
                if s["y"] is not None:
                    df["label"] = np.array(self.data.class_names)[s["y"][:SAMPLE_CAP_ROWS]]
                df.to_csv(d / f"{primary}_rows.csv", index=False)
            if s["augment"] is not None and mod != "image":
                pass
        elif t == "imputation":
            rep, model = self.reps[primary], self.models[primary]
            X = self.data.X.iloc[:SAMPLE_CAP_ROWS].reset_index(drop=True)
            Z, M = rep.transform(X), rep.observed_mask(X)
            imputed = rep.inverse(model.impute(Z, M.astype(np.float32)))
            imputed["n_values_imputed"] = X.isna().sum(axis=1).to_numpy()
            imputed.to_csv(d / f"{primary}_imputed_rows.csv", index=False)
        elif t == "anomaly_detection":
            order = np.argsort(-self.anomaly_scores[primary])[:100]
            pd.DataFrame({"test_row": self.splits.test.index[order],
                          "anomaly_score": self.anomaly_scores[primary][order],
                          "is_labelled_anomaly": self.anomaly_y[order]}) \
                .to_csv(d / f"{primary}_top100_flagged.csv", index=False)

    def _manifest(self) -> dict:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                 text=True, timeout=10).stdout.strip()
        except Exception:
            sha = None
        pkgs = {p: _version(p) for p in ("torch", "numpy", "scikit-learn", "scanpy", "pandas",
                                         "claude-agent-sdk")}
        return {"command": " ".join(sys.argv), "data": str(self.data_path),
                "objective": self.objective, "created": datetime.now().isoformat(timespec="seconds"),
                "seconds_total": round(time.perf_counter() - self.t0, 1),
                "config": self.cfg.model_dump(), "hardware": self.hw.to_record(),
                "python": platform.python_version(), "packages": pkgs, "git_sha": sha,
                "run_dir": str(self.run_dir), "plan_source": self.plan.source,
                "llm": {"enabled": self.client.available, "model_requested": self.client.model,
                        "calls": self.client.calls,
                        "total_cost_usd": round(sum(c.get("cost_usd") or 0 for c in self.client.calls
                                                    if not c.get("cached")), 4),
                        "replayed": bool(self.cfg.replay)}}


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _merge(results: dict, new: dict) -> dict:
    """Overlay per-model entries of ``new`` onto a deep copy of ``results``."""
    merged = json.loads(json.dumps(results, default=_jsonable))
    new = json.loads(json.dumps(new, default=_jsonable))
    for key, block in new.items():
        if key == "generative":
            g = merged.setdefault("generative", {})
            g.setdefault("models", {}).update(block.get("models", {}))
            if "utility" in block:
                g.setdefault("utility", block["utility"])
                g["utility"].setdefault("models", {}).update(block["utility"]["models"])
        elif key in ("imputation", "anomaly"):
            merged.setdefault(key, {}).setdefault("methods", {}).update(block["methods"])
        else:
            merged.setdefault(key, {}).update(block)
    return merged


def run(data_path, objective: str, out_dir, **overrides) -> dict:
    """Run the agent; see ``genmodel_agent.run_agent``."""
    return Agent(data_path, objective, out_dir, **overrides).run()
