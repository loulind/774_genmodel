"""The one place the agent talks to a language model (adapted from Project 1).

Planner, critic and report writer all call ``LLMClient.complete(role, system,
user)``. Which model answers is decided here:

    claude_agent_sdk   drives the local `claude` CLI with its existing login
                       (no API key). Tools are disabled and the user's own
                       settings are not loaded, so the call is a plain
                       text-in/text-out completion.
    offline            always unavailable; callers fall back to the rule-based
                       planner/critic/writer. ``--no-llm`` selects it.

Reproducibility. Every response is written to ``<run>/llm_cache/<role>_<hash>.json``
with the exact prompt, the model requested, the model that actually answered
(read back from the SDK, so a renamed or substituted model is visible), timing
and cost. ``--replay <dir>`` serves responses from such a cache and never calls a
model: a run can be reproduced decision-for-decision without LLM access.

Limits. Output tokens are capped per call (CLAUDE_CODE_MAX_OUTPUT_TOKENS) and so
is spend (max_budget_usd); both are recorded in the cache entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"


class LLMUnavailable(RuntimeError):
    """No model could be reached (or the replay cache lacks this prompt)."""


@dataclass
class LLMResponse:
    text: str
    backend: str
    model_requested: str
    model_used: str
    seconds: float
    cost_usd: float | None = None
    cached: bool = False

    def to_record(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d["response_chars"] = len(self.text)
        return d


def _ensure_cli_on_path() -> None:
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")


class ClaudeAgentSDKBackend:
    name = "claude_agent_sdk"

    def __init__(self, model: str = DEFAULT_MODEL, max_output_tokens: int = 8000,
                 max_budget_usd: float = 2.0):
        _ensure_cli_on_path()
        if shutil.which("claude") is None:
            raise LLMUnavailable("the `claude` CLI is not on PATH "
                                 "(install: curl -fsSL https://claude.ai/install.sh | bash)")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_budget_usd = max_budget_usd

    def complete(self, system: str, user: str) -> LLMResponse:
        import anyio
        from claude_agent_sdk import ClaudeAgentOptions, query

        async def _run():
            options = ClaudeAgentOptions(
                system_prompt=system,
                model=self.model,
                tools=[],                     # no tools: the agent's tools are our Python
                setting_sources=[],           # ignore the user's CLAUDE.md / settings
                max_turns=3,
                max_budget_usd=self.max_budget_usd,
                env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_output_tokens)},
            )
            chunks, cost, used = [], None, None
            async for message in query(prompt=user, options=options):
                kind = type(message).__name__
                if kind == "AssistantMessage":
                    used = getattr(message, "model", None) or used
                    for block in getattr(message, "content", []) or []:
                        if getattr(block, "text", None):
                            chunks.append(block.text)
                elif kind == "ResultMessage":
                    cost = getattr(message, "total_cost_usd", None)
                    if getattr(message, "is_error", False):
                        raise LLMUnavailable(f"CLI reported an error: "
                                             f"{getattr(message, 'result', '')!s:.200}")
            return "".join(chunks), cost, used

        start = time.perf_counter()
        try:
            text, cost, used = anyio.run(_run)
        except LLMUnavailable:
            raise
        except Exception as err:
            raise LLMUnavailable(f"claude_agent_sdk failed: {err}") from err
        if not text.strip():
            raise LLMUnavailable("claude_agent_sdk returned no text")
        return LLMResponse(text=text, backend=self.name, model_requested=self.model,
                           model_used=used or "unknown", seconds=time.perf_counter() - start,
                           cost_usd=cost)


class OfflineBackend:
    name = "offline"
    model = "none"

    def complete(self, system: str, user: str) -> LLMResponse:
        raise LLMUnavailable("offline: --no-llm selected or no backend available")


class LLMClient:
    """Backend + per-run disk cache (+ optional replay-only mode)."""

    def __init__(self, backend, cache_dir: str | Path, replay_from: str | Path | None = None):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.replay_from = Path(replay_from) if replay_from else None
        self.calls: list[dict] = []

    @property
    def available(self) -> bool:
        return self.replay_from is not None or not isinstance(self.backend, OfflineBackend)

    @property
    def model(self) -> str:
        return getattr(self.backend, "model", "none")

    def _key(self, role: str, system: str, user: str) -> str:
        blob = json.dumps({"model": self.model, "system": system, "user": user}, sort_keys=True)
        return f"{role}_{hashlib.sha256(blob.encode()).hexdigest()[:16]}.json"

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        key = self._key(role, system, user)
        for directory in [d for d in (self.replay_from, self.cache_dir) if d is not None]:
            path = directory / key
            if path.exists():
                rec = json.loads(path.read_text())
                resp = LLMResponse(text=rec["text"], backend=rec["backend"],
                                   model_requested=rec["model_requested"],
                                   model_used=rec["model_used"], seconds=0.0,
                                   cost_usd=rec.get("cost_usd"), cached=True)
                self._record(role, key, resp)
                if directory != self.cache_dir:          # keep this run's cache complete
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / key).write_text(path.read_text())
                return resp
        if self.replay_from is not None:
            # Prompts embed run results, so a replay can differ in incidental
            # digits; the call sequence (planner, critic, writer) is what is
            # reproduced, so fall back to the cached response for this role.
            same_role = sorted(self.replay_from.glob(f"{role}_*.json"))
            if len(same_role) == 1:
                rec = json.loads(same_role[0].read_text())
                resp = LLMResponse(text=rec["text"], backend=rec["backend"],
                                   model_requested=rec["model_requested"],
                                   model_used=rec["model_used"], seconds=0.0,
                                   cost_usd=rec.get("cost_usd"), cached=True)
                self._record(role, same_role[0].name + " (matched by role)", resp)
                return resp
            raise LLMUnavailable(f"replay: no cached response for {role} ({key})")
        resp = self.backend.complete(system, user)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / key).write_text(json.dumps({
            "role": role, "backend": resp.backend, "model_requested": resp.model_requested,
            "model_used": resp.model_used, "seconds": round(resp.seconds, 2),
            "cost_usd": resp.cost_usd,
            "max_output_tokens": getattr(self.backend, "max_output_tokens", None),
            "system": system, "user": user, "text": resp.text}, indent=2))
        self._record(role, key, resp)
        return resp

    def _record(self, role, key, resp):
        self.calls.append({"role": role, "cache_file": key, **resp.to_record()})


def make_client(enabled: bool, model: str, cache_dir, replay_from=None,
                max_output_tokens: int = 8000, max_budget_usd: float = 2.0) -> LLMClient:
    if replay_from:
        backend = OfflineBackend()
        backend.model = model                 # cache keys include the model name
        return LLMClient(backend, cache_dir, replay_from)
    if not enabled:
        return LLMClient(OfflineBackend(), cache_dir)
    try:
        backend = ClaudeAgentSDKBackend(model, max_output_tokens, max_budget_usd)
    except LLMUnavailable as err:
        print(f"  LLM unavailable ({err}); using rule-based planner, critic and writer")
        backend = OfflineBackend()
    return LLMClient(backend, cache_dir)


# ---------------------------------------------------------------------------
# Response parsing (from Project 1)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response: fenced blocks first, then the
    outermost brace span; several partial objects are merged (later keys win)."""
    candidates = [m.strip() for m in _FENCE.findall(text)]
    if text.count("```") == 1:
        opening = re.search(r"```(?:json)?\s*", text)
        if opening:
            candidates.append(text[opening.end():].strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    parsed, first_error = [], None
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError as err:
            first_error = first_error or f"{err.msg} at line {err.lineno}"
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    if len(parsed) == 1:
        return parsed[0]
    if parsed:
        merged: dict = {}
        for obj in parsed:
            merged.update(obj)
        return merged
    raise ValueError(f"no parseable JSON object in response"
                     f"{' (' + first_error + ')' if first_error else ''}; "
                     f"starts: {text[:200]!r}")
