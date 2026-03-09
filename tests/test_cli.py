"""Tests for the CLI — validate command, error paths, and open/replay output."""

import json
import tempfile
from pathlib import Path

import pytest

from tracer.schema import Step, Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_trace(path: Path, trace: Trace) -> None:
    trace.save(path)


def _good_trace() -> Trace:
    return Trace(
        run_id="abc123",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={"messages": [{"role": "user", "content": "Hello"}]},
                output={"text": "Hi there"},
                parameters={"temperature": 0.0},
                timestamp="2025-01-01T00:00:00+00:00",
            )
        ],
        final_output="Hi there",
    )


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------

def test_validate_valid_file(capsys):
    from tracer.cli import cmd_validate
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ok.rpack"
        _write_trace(path, _good_trace())

        args = argparse.Namespace(rpack=str(path))
        cmd_validate(args)

    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_validate_invalid_schema_exits_nonzero(capsys):
    from tracer.cli import cmd_validate
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.rpack"
        # Write a trace with a step missing the 'messages' key.
        bad = Trace(
            run_id="x",
            provider="openai",
            steps=[
                Step(step=1, model="", input={"wrong": []}, output={"no_text": ""})
            ],
        )
        _write_trace(path, bad)

        args = argparse.Namespace(rpack=str(path))
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out


def test_validate_missing_file_exits_nonzero(capsys):
    from tracer.cli import cmd_validate
    import argparse

    args = argparse.Namespace(rpack="/nonexistent/path.rpack")
    with pytest.raises(SystemExit) as exc_info:
        cmd_validate(args)

    assert exc_info.value.code == 1


def test_validate_corrupt_json_exits_nonzero(capsys):
    from tracer.cli import cmd_validate
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corrupt.rpack"
        path.write_text("{ not valid json", encoding="utf-8")

        args = argparse.Namespace(rpack=str(path))
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# open command
# ---------------------------------------------------------------------------

def test_open_prints_run_id(capsys):
    from tracer.cli import cmd_open
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.rpack"
        _write_trace(path, _good_trace())

        args = argparse.Namespace(rpack=str(path))
        cmd_open(args)

    captured = capsys.readouterr()
    assert "abc123" in captured.out
    assert "openai" in captured.out
    assert "Step 1" in captured.out


def test_open_shows_tools_in_input(capsys):
    """When a step has tools in input, open should name them in the summary line."""
    from tracer.cli import cmd_open
    import argparse

    trace = Trace(
        run_id="tooltest",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
                output={"text": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
            )
        ],
        final_output="[tool calls: get_weather]",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.rpack"
        _write_trace(path, trace)
        args = argparse.Namespace(rpack=str(path))
        cmd_open(args)

    captured = capsys.readouterr()
    assert "get_weather" in captured.out
    assert "tool(s)" in captured.out


def test_open_truncates_long_content(capsys):
    from tracer.cli import cmd_open
    import argparse

    long_content = "x" * 500
    trace = Trace(
        run_id="trunc",
        provider="openai",
        steps=[
            Step(
                step=1,
                model="gpt-4",
                input={"messages": [{"role": "user", "content": long_content}]},
                output={"text": long_content},
            )
        ],
        final_output=long_content,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.rpack"
        _write_trace(path, trace)

        args = argparse.Namespace(rpack=str(path))
        cmd_open(args)

    captured = capsys.readouterr()
    assert "more chars" in captured.out
    assert long_content not in captured.out


# ---------------------------------------------------------------------------
# replay command
# ---------------------------------------------------------------------------

def test_replay_prints_step(capsys):
    from tracer.cli import cmd_replay
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.rpack"
        _write_trace(path, _good_trace())

        args = argparse.Namespace(rpack=str(path), step=1)
        cmd_replay(args)

    captured = capsys.readouterr()
    assert "Step 1" in captured.out
    assert "gpt-4" in captured.out


def test_replay_bad_step_exits_nonzero(capsys):
    from tracer.cli import cmd_replay
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.rpack"
        _write_trace(path, _good_trace())

        args = argparse.Namespace(rpack=str(path), step=99)
        with pytest.raises(SystemExit) as exc_info:
            cmd_replay(args)

    assert exc_info.value.code == 1
