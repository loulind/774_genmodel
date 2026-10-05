"""Generative modeling agent.

Public entry point::

    from genmodel_agent import run_agent
    result = run_agent("data/pathmnist.npz", "generate class-conditional ...",
                       "outputs/my_run", quick=True)
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def run_agent(data_path: str | Path, objective: str, out_dir: str | Path,
              **config_overrides):
    """Run the full agent loop on one dataset and objective.

    Returns a dict summarising the run (decisions, metrics, output paths).
    Implemented by the orchestrator; imported lazily so `import genmodel_agent`
    stays cheap for callers that only want the schema or the profiler.
    """
    from .orchestrator import run  # noqa: WPS433  (lazy by design)
    return run(data_path, objective, out_dir, **config_overrides)


__all__ = ["run_agent", "__version__"]
