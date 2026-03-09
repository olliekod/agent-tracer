import json
import tempfile
from pathlib import Path

import pytest

from tracer.schema import Step, Trace


def test_step_roundtrip():
    step = Step(
        step=1,
        model="gpt-4",
        input={"messages": [{"role": "user", "content": "hello"}]},
        output={"text": "hi there"},
        parameters={"temperature": 0.5},
        timestamp="2025-01-01T00:00:00+00:00",
    )
    data = step.to_dict()
    restored = Step.from_dict(data)
    assert restored.step == 1
    assert restored.model == "gpt-4"
    assert restored.input == step.input
    assert restored.output == step.output
    assert restored.parameters == step.parameters


def test_trace_save_load():
    trace = Trace(
        run_id="abc123",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={"messages": [{"role": "user", "content": "hello"}]},
                output={"text": "hi"},
                parameters={},
                timestamp="2025-01-01T00:00:00+00:00",
            )
        ],
        final_output="hi",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        trace.save(path)

        loaded = Trace.load(path)
        assert loaded.run_id == "abc123"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].output == {"text": "hi"}

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["run_id"] == "abc123"


def test_trace_json_structure():
    trace = Trace(run_id="x", provider="openai")
    data = trace.to_dict()
    assert set(data.keys()) == {"run_id", "provider", "steps", "final_output"}
    json.loads(json.dumps(data))


def test_step_missing_optional_fields():
    data = {
        "step": 1,
        "model": "gpt-4",
        "input": {"messages": []},
        "output": {"text": ""},
    }
    step = Step.from_dict(data)
    assert step.parameters == {}
    assert step.timestamp == ""


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------

def test_step_from_dict_missing_required_raises():
    with pytest.raises(ValueError, match="missing required field"):
        Step.from_dict({"step": 1, "model": "gpt-4"})  # missing input and output


def test_step_from_dict_missing_single_field_names_it():
    with pytest.raises(ValueError, match="output"):
        Step.from_dict(
            {"step": 1, "model": "gpt-4", "input": {"messages": []}}
        )


def test_trace_from_dict_missing_required_raises():
    with pytest.raises(ValueError, match="missing required field"):
        Trace.from_dict({"provider": "openai"})  # missing run_id


def test_trace_load_invalid_json_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.rpack"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            Trace.load(path)


def test_trace_validate_returns_empty_for_valid_trace():
    trace = Trace(
        run_id="abc",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={"messages": []},
                output={"text": "ok"},
            )
        ],
    )
    assert trace.validate() == []


def test_trace_validate_catches_empty_run_id():
    trace = Trace(run_id="", provider="openai")
    errors = trace.validate()
    assert any("run_id" in e for e in errors)


def test_trace_validate_catches_missing_messages_key():
    trace = Trace(
        run_id="abc",
        provider="openai",
        steps=[
            Step(step=1, model="gpt-4", input={"not_messages": []}, output={"text": ""})
        ],
    )
    errors = trace.validate()
    assert any("messages" in e for e in errors)


def test_trace_validate_catches_missing_text_key():
    trace = Trace(
        run_id="abc",
        provider="openai",
        steps=[
            Step(step=1, model="gpt-4", input={"messages": []}, output={"no_text": ""})
        ],
    )
    errors = trace.validate()
    assert any("text" in e for e in errors)


def test_trace_validate_catches_empty_model():
    trace = Trace(
        run_id="abc",
        provider="openai",
        steps=[
            Step(step=1, model="", input={"messages": []}, output={"text": ""})
        ],
    )
    errors = trace.validate()
    assert any("model" in e for e in errors)
