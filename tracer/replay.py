"""Replay engine — reads .rpack files and simulates recorded steps.

No LLM calls are made. Replay is fully deterministic.
"""

from __future__ import annotations

from typing import Any

from tracer.schema import Step, Trace


def load_trace(path: str) -> Trace:
    """Load a trace from an .rpack file."""
    return Trace.load(path)


def get_step(trace: Trace, step_number: int) -> Step:
    """Return a specific step by number (1-indexed)."""
    for step in trace.steps:
        if step.step == step_number:
            return step
    raise ValueError(
        f"Step {step_number} not found (trace has {len(trace.steps)} steps)"
    )


def replay_step(trace: Trace, step_number: int) -> dict[str, Any]:
    """Replay a single step — returns stored input and output verbatim."""
    step = get_step(trace, step_number)
    return {
        "step": step.step,
        "model": step.model,
        "input": step.input,
        "output": step.output,
        "parameters": step.parameters,
        "timestamp": step.timestamp,
    }
