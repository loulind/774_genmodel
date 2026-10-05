"""Per-run decision log: every choice, why, and what was rejected.

The report's "how the agent decides" section is built from these files, so a
decision that is not logged here effectively did not happen. Entries are
append-only and written to ``decision_log.json`` after every step, so even a run
that crashes leaves a record of how far it got and why.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Alternative, Decision


class DecisionLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[Decision] = []

    def add(self, step: str, decision: str, chosen: str, rationale: str,
            alternatives: list[tuple[str, str]] | None = None,
            evidence: dict | None = None, source: str = "heuristic") -> Decision:
        entry = Decision(
            step=step, decision=decision, chosen=chosen, rationale=rationale,
            alternatives_rejected=[Alternative(option=o, reason=r) for o, r in alternatives or []],
            evidence=evidence or {}, source=source,
        )
        self.entries.append(entry)
        self.flush()
        return entry

    def extend(self, entries: list[Decision]) -> None:
        self.entries.extend(entries)
        self.flush()

    def flush(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([e.model_dump() for e in self.entries], indent=2))

    def by_step(self, step: str) -> list[Decision]:
        return [e for e in self.entries if e.step == step]
