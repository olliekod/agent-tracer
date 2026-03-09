import tempfile
from pathlib import Path

from tracer.recorder import Recorder
from tracer.schema import Trace


def test_recorder_writes_rpack():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        rec = Recorder(output_path=str(path))
        rec.record_step(
            model="gpt-4",
            input_data={"messages": [{"role": "user", "content": "test"}]},
            output_data={"text": "response"},
            parameters={"temperature": 0.7},
        )
        rec.flush()

        trace = Trace.load(path)
        assert len(trace.steps) == 1
        assert trace.steps[0].model == "gpt-4"
        assert trace.steps[0].output == {"text": "response"}
        assert trace.final_output == "response"


def test_recorder_increments_steps():
    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(output_path=str(Path(tmp) / "test.rpack"))
        rec.record_step("gpt-4", {"messages": []}, {"text": "a"})
        rec.record_step("gpt-4", {"messages": []}, {"text": "b"})
        assert rec.trace.steps[0].step == 1
        assert rec.trace.steps[1].step == 2
        rec.flush()


def test_recorder_sets_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(output_path=str(Path(tmp) / "test.rpack"))
        rec.record_step("gpt-4", {"messages": []}, {"text": "x"})
        assert rec.trace.steps[0].timestamp != ""
        rec.flush()


def test_flush_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        rec = Recorder(output_path=str(path))
        rec.record_step("gpt-4", {"messages": []}, {"text": "x"})
        rec.flush()
        rec.flush()
        trace = Trace.load(path)
        assert len(trace.steps) == 1


def test_recorder_captures_tool_calls():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        rec = Recorder(output_path=str(path))
        rec.record_step(
            model="gpt-4",
            input_data={
                "messages": [
                    {"role": "user", "content": "What's the weather?"},
                ]
            },
            output_data={
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
        )
        rec.flush()

        trace = Trace.load(path)
        step = trace.steps[0]
        assert step.output["tool_calls"][0]["function"]["name"] == "get_weather"


def test_default_output_path_is_timestamped():
    rec = Recorder()
    assert rec.output_path.startswith("trace_")
    assert rec.output_path.endswith(".rpack")
    rec._flushed = True  # prevent atexit from writing


def test_final_output_uses_tool_call_description_when_last_step_has_no_text():
    """When the last step has only tool calls and no text, final_output should
    describe the tool calls rather than being an empty string."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        rec = Recorder(output_path=str(path))
        rec.record_step(
            model="gpt-4",
            input_data={"messages": [{"role": "user", "content": "What's the weather?"}]},
            output_data={
                "text": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
        )
        rec.flush()

        trace = Trace.load(path)
        assert "get_weather" in trace.final_output


def test_final_output_falls_back_to_last_text_step():
    """When the last step has no text or tool calls, walk back to find text."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.rpack"
        rec = Recorder(output_path=str(path))
        rec.record_step("gpt-4", {"messages": []}, {"text": "intermediate answer"})
        rec.record_step("gpt-4", {"messages": []}, {"text": ""})  # last step empty
        rec.flush()

        trace = Trace.load(path)
        assert trace.final_output == "intermediate answer"
