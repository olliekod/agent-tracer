import tempfile
from pathlib import Path

import pytest

from tracer.replay import get_step, load_trace, replay_step
from tracer.schema import Step, Trace


def _make_trace() -> Trace:
    return Trace(
        run_id="test",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={"messages": [{"role": "user", "content": "hello"}]},
                output={"text": "hi"},
                parameters={"temperature": 0.2},
                timestamp="2025-01-01T00:00:00+00:00",
            ),
            Step(
                step=2,
                model="gpt-4",
                input={"messages": [{"role": "user", "content": "bye"}]},
                output={"text": "goodbye"},
                parameters={},
                timestamp="2025-01-01T00:01:00+00:00",
            ),
        ],
        final_output="goodbye",
    )


def test_get_step():
    trace = _make_trace()
    step = get_step(trace, 1)
    assert step.model == "gpt-4"
    assert step.output == {"text": "hi"}


def test_get_step_not_found():
    trace = _make_trace()
    with pytest.raises(ValueError, match="Step 5 not found"):
        get_step(trace, 5)


def test_replay_step_returns_stored_data():
    trace = _make_trace()
    result = replay_step(trace, 2)
    assert result["step"] == 2
    assert result["output"] == {"text": "goodbye"}
    assert result["input"]["messages"][0]["content"] == "bye"


def test_load_trace_from_file():
    trace = _make_trace()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        trace.save(path)
        loaded = load_trace(str(path))
        assert loaded.run_id == "test"
        assert len(loaded.steps) == 2


def test_replay_step_with_tool_calls():
    trace = Trace(
        run_id="tools",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={
                    "messages": [
                        {"role": "user", "content": "What's the weather?"},
                    ]
                },
                output={
                    "text": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                parameters={},
                timestamp="2025-01-01T00:00:00+00:00",
            ),
            Step(
                step=2,
                model="gpt-4",
                input={
                    "messages": [
                        {"role": "user", "content": "What's the weather?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "content": "Sunny, 22°C",
                            "tool_call_id": "call_abc",
                        },
                    ]
                },
                output={"text": "The weather in Paris is sunny at 22°C."},
                parameters={},
                timestamp="2025-01-01T00:00:01+00:00",
            ),
        ],
        final_output="The weather in Paris is sunny at 22°C.",
    )

    result = replay_step(trace, 1)
    assert result["output"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["output"]["text"] == ""

    result2 = replay_step(trace, 2)
    assert result2["output"]["text"] == "The weather in Paris is sunny at 22°C."
    assert len(result2["input"]["messages"]) == 3
    assert result2["input"]["messages"][2]["role"] == "tool"
