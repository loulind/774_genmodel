# Generative modeling agent

An AI agent that takes a dataset and a free-text objective and decides **whether and how** to apply generative modeling. In one command it:

1. profiles the data;
2. interprets the objective;
3. decides whether a generative model is warranted, and declines with an alternative when it is not;
4. chooses models and hyperparameters;
5. trains them and generates samples;
6. evaluates fidelity, diversity, memorization and downstream utility, or the task-specific metric;
7. lets a critic order at most one revision;
8. recommends downstream uses and warns against misuses;
9. writes a report.

Every decision is logged with its rationale and the alternatives it rejected (`decision_log.json`).

BIOS 774, Project 2.

```
load -> profile -> parse objective -> plan (LLM | rules) -> validate/repair/fallback
     -> split -> train (time/epoch caps) -> sample -> evaluate (vs real-holdout reference)
     -> critic (<= 1 revision, kept only if the primary metric improves)
     -> recommendations -> report (Markdown + HTML)
```

**Division of labour.** The LLM (Claude, through the local `claude` CLI) acts only as planner, critic and report writer, and returns JSON. Deterministic Python does the profiling, plan validation, training and evaluation. A rule-based planner, critic and writer cover the whole pipeline when no LLM is available.

**Model library.**
- VAE family: Gaussian/categorical VAE (optionally conditional), NB and ZINB count VAEs (scVI-style), and a MIWAE masked VAE for imputation.
- GANs: WGAN-GP (optionally conditional) and a CTGAN-style tabular GAN.
- Normalizing flows: RealNVP and MAF.
- Diffusion: DDPM with a small U-Net for images and an MLP denoiser for vectors (cosine schedule, v-prediction, DDIM sampling, classifier-free guidance).

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is optional: CUDA or Apple MPS is detected automatically, and training budgets are scaled to the device.

```bash
uv sync
```

If you don't use uv, the pinned requirements work with plain pip:

```bash
pip install -r requirements.txt && pip install -e .
```

**Optional LLM.** Install the `claude` CLI and log in (`curl -fsSL https://claude.ai/install.sh | bash`, then `claude`). Without it, or with `--no-llm`, the rule-based components are used.

## Data

```bash
uv run python scripts/prepare_data.py
```

This downloads every dataset into `data/` (gitignored), checking MD5 where published:

- PathMNIST and BreastMNIST (28×28), plus a class-name sidecar for each;
- PBMC 3k raw counts, with scanpy's published cell-type labels joined by cell barcode;
- UCI Chronic Kidney Disease (OpenML #42972);
- Mammography (OpenML #310);
- Breast Cancer Wisconsin (scikit-learn).

## Run: one command per dataset

```bash
# generated_report_1: PathMNIST, low-label augmentation (~40 min on an M2 Pro)
uv run genagent run --data data/pathmnist.npz --out outputs/generated_report_1 \
  --objective "Only a small labelled subset (~5% of training patches) is available. Generate class-conditional synthetic tissue patches, assess whether augmenting the subset improves a downstream classifier, and whether a classifier trained on synthetic data alone transfers to real data."

# generated_report_2: PBMC 3k, rare cell types (~5 min)
uv run genagent run --data data/pbmc3k_labeled.h5ad --out outputs/generated_report_2 \
  --objective "Generate realistic synthetic cells for rare cell types to support downstream cell-type classification."
```

Options:

| flag | effect |
|---|---|
| `--quick` | smoke test: about 1 minute per model, ≤3 epochs, ≤2,000 training samples |
| `--no-llm` | rule-based planner, critic and writer; no LLM access needed |
| `--replay runs/<run>/llm_cache` | reproduce a previous run's LLM decisions from its cached responses |
| `--config my.yaml` | budgets, seed and LLM settings (see `configs/default.yaml`); never model choices |
| `--label-column`, `--device`, `--seed`, `--model` | label column; `auto`/`cuda`/`mps`/`cpu`; random seed; LLM model id |

`uv run genagent hardware` prints the detected device and the time budgets.

**From Python.** `run_agent` is the entry point for other agents:

```python
from genmodel_agent import run_agent
result = run_agent("data/mammography.csv", "Detect anomalous findings", "outputs/x", quick=True)
result["recommendations"]["verdict"]
```

## Outputs (per run)

| file | contents |
|---|---|
| `report.md`, `report.html` | final analysis report |
| `decision_log.json` | every decision: step, choice, rationale, rejected alternatives, evidence, source (heuristic / llm / validator / critic) |
| `profile.json`, `plan.json` | data profile; the validated plan that ran |
| `evaluation.json`, `tables/*.csv` | metrics, including a real-holdout reference row and bootstrap intervals |
| `model_summary.json` | architectures, hyperparameters, training history, critic verdict, revision |
| `recommendations.json` | downstream uses and misuse warnings |
| `figures/` | sample grids, real-vs-synthetic embeddings, gene or marginal statistics, utility, training curves |
| `synthetic_sample/` | capped sample of generated data (or imputed rows / flagged anomalies) |
| `run_manifest.json` | command, config, hardware, package versions, git commit, LLM calls and cost |

Scratch files (`runs/<timestamp>_<name>/`: log, LLM cache) are gitignored.

## Robustness check

`outputs/robustness/summary.md` compares the agent's decisions across six dataset/objective pairs. The robustness rows come from these `--quick` runs (artifacts deleted after summarizing):

```bash
uv run genagent run --quick --data data/breastmnist.npz --out tmp/breastmnist --objective "Generate synthetic images of the under-represented class to rebalance the training set for a downstream classifier."
uv run genagent run --quick --data data/chronic_kidney_disease.csv --out tmp/ckd --objective "Impute the missing clinical measurements so the full cohort can be analysed."
uv run genagent run --quick --data data/mammography.csv --out tmp/mammography --objective "Detect anomalous mammography findings among mostly normal cases."
uv run genagent run --quick --data data/breast_cancer_wisconsin.csv --out tmp/wisconsin --objective "Generate synthetic patients to augment the training data and improve a malignancy classifier."
uv run python scripts/summarize_robustness.py PathMNIST=outputs/generated_report_1:full "PBMC 3k=outputs/generated_report_2:full" BreastMNIST=tmp/breastmnist "Chronic Kidney Disease=tmp/ckd" Mammography=tmp/mammography "Breast Cancer Wisconsin=tmp/wisconsin"
```

## Tests

```bash
uv run pytest -q
```

The suite takes about 1.5 minutes on CPU. It covers known-answer tests for every model family and metric, planner and validator guardrails, LLM fallback paths (with a fake backend), and end-to-end runs. It also includes a syntax-tree check that no dataset name appears in the package's code.

## Reproducibility notes

- **Seeds.** All seeds are fixed (`seed: 774`). Splits, subsamples and decisions are exactly reproducible. GPU kernels (MPS especially) are not bit-for-bit deterministic, so retrained models agree only to floating-point noise.
- **LLM planning.** The LLM planner is not deterministic across *fresh* runs; two PathMNIST runs chose guidance 1.0 and 1.5. Each run caches every prompt and response, and `--replay` reproduces that run's decisions exactly.
- **Pinned versions.** `pyproject.toml` with `uv.lock`, and an exported `requirements.txt`.

## Layout

```
genmodel_agent/
  __init__.py      run_agent()
  cli.py           genagent command
  orchestrator.py  the agent loop
  config.py        budgets and LLM settings
  hardware.py      device detection, seeding
  data/            loaders, download, splits
  profiler.py      modality, shape, missingness, count diagnostics (NB vs ZINB by MLE), duplicates
  objective.py     free-text objective -> TaskSpec
  planner/         heuristic.py (rules), llm.py (LLM + repair + fallback), validator.py (guardrails)
  models/          registry.py, vae.py, gan.py, flows.py, diffusion.py, train.py
  preprocess.py    representations with inverse transforms
  evaluate/        fidelity, memorization, utility, task-specific metrics, feature spaces
  critic.py        triggers + one revision
  recommend.py     downstream uses and misuse warnings
  report.py        report and tables
  figures.py       figures
  llm.py           claude CLI backend, cache, replay
prompts/           planner.md, critic.md, writer.md
configs/           default.yaml
scripts/           prepare_data.py, summarize_robustness.py
tests/
outputs/           generated_report_1, generated_report_2, robustness
```
