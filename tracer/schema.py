"""Provider-agnostic trace schema for .rpack artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Step:
    """A single LLM interaction."""

    step: int
    model: str
    input: dict[str, Any]
    output: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model": self.model,
            "input": self.input,
            "parameters": self.parameters,
            "output": self.output,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        required = ("step", "model", "input", "output")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"Step is missing required field(s): {', '.join(missing)}"
            )
        return cls(
            step=data["step"],
            model=data["model"],
            input=data["input"],
            output=data["output"],
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class Trace:
    """A complete agent run trace stored as an .rpack artifact."""

    run_id: str
    provider: str
    steps: list[Step] = field(default_factory=list)
    final_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "steps": [s.to_dict() for s in self.steps],
            "final_output": self.final_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        required = ("run_id", "provider")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"Trace is missing required field(s): {', '.join(missing)}"
            )
        return cls(
            run_id=data["run_id"],
            provider=data["provider"],
            steps=[Step.from_dict(s) for s in data.get("steps", [])],
            final_output=data.get("final_output", ""),
        )

    def validate(self) -> list[str]:
        """Return a list of validation errors. Empty list means the trace is valid."""
        errors: list[str] = []

        if not self.run_id:
            errors.append("run_id is empty")
        if not self.provider:
            errors.append("provider is empty")

        for i, step in enumerate(self.steps):
            prefix = f"step[{i + 1}]"
            if not step.model:
                errors.append(f"{prefix}: model is empty")
            if "messages" not in step.input:
                errors.append(f"{prefix}: input is missing 'messages' key")
            if "text" not in step.output:
                errors.append(f"{prefix}: output is missing 'text' key")

        return errors

    def save(self, path: str | Path) -> None:
        """Write trace to an .rpack file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Read trace from an .rpack file."""
        raw = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        return cls.from_dict(data)
