"""Records LLM interactions into a trace artifact."""

from __future__ import annotations

import atexit
import uuid
from datetime import datetime, timezone
from typing import Any

from tracer.schema import Step, Trace


def _default_output_path() -> str:
    return datetime.now().strftime("trace_%Y-%m-%d_%H-%M-%S.rpack")


def _resolve_final_output(steps: list) -> str:
    """Derive final_output from the last step.

    Prefers non-empty text. Falls back to a tool-call description when the
    last step only produced tool calls (e.g. the agent is mid-loop).
    """
    last = steps[-1]
    text = last.output.get("text", "")
    if text:
        return text
    tool_calls = last.output.get("tool_calls", [])
    if tool_calls:
        names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        return f"[tool calls: {', '.join(names)}]"
    # Walk backwards for the last step that produced text.
    for step in reversed(steps[:-1]):
        t = step.output.get("text", "")
        if t:
            return t
    return ""


class Recorder:
    """Collects LLM interaction events and writes them to an .rpack file."""

    def __init__(self, output_path: str | None = None, provider: str = "openai"):
        self.trace = Trace(
            run_id=uuid.uuid4().hex[:12],
            provider=provider,
        )
        self.output_path = output_path or _default_output_path()
        self._step_counter = 0
        self._flushed = False
        atexit.register(self.flush)

    def record_step(
        self,
        model: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Append a normalized LLM interaction step."""
        self._step_counter += 1
        step = Step(
            step=self._step_counter,
            model=model,
            input=input_data,
            output=output_data,
            parameters=parameters or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.trace.steps.append(step)

    def flush(self) -> None:
        """Write the trace to disk. Safe to call multiple times."""
        if self._flushed:
            return
        self._flushed = True
        if self.trace.steps:
            self.trace.final_output = _resolve_final_output(self.trace.steps)
        self.trace.save(self.output_path)


# ---------------------------------------------------------------------------
# Global recorder used by adapters
# ---------------------------------------------------------------------------

_recorder: Recorder | None = None


def get_recorder() -> Recorder:
    """Return the active global recorder, creating one if needed."""
    global _recorder
    if _recorder is None:
        _recorder = Recorder()
    return _recorder


def init_recorder(output_path: str | None = None, provider: str = "openai") -> Recorder:
    """Initialize and return the global recorder."""
    global _recorder
    _recorder = Recorder(output_path=output_path, provider=provider)
    return _recorder
