"""Tests for the Anthropic adapter: patch/unpatch idempotency, streaming warning,
response normalization, system prompt folding, and tool call capture."""

import json
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build fake Anthropic response objects
# ---------------------------------------------------------------------------

def _make_text_block(text="Hello!"):
    return SimpleNamespace(type="text", text=text)


def _make_tool_use_block(name="get_weather", input_dict=None, block_id="toolu_1"):
    return SimpleNamespace(
        type="tool_use",
        id=block_id,
        name=name,
        input=input_dict or {"city": "Paris"},
    )


def _make_response(blocks=None, model="claude-3-5-haiku-20241022"):
    return SimpleNamespace(content=blocks or [_make_text_block()], model=model)


# ---------------------------------------------------------------------------
# Patch / unpatch idempotency
# ---------------------------------------------------------------------------

def test_patch_is_idempotent():
    pytest.importorskip("anthropic")
    from tracer.adapters import anthropic_adapter
    try:
        from anthropic.resources.messages.messages import Messages
    except ImportError:
        from anthropic.resources.messages import Messages

    if anthropic_adapter._patched:
        anthropic_adapter.unpatch()

    anthropic_adapter.patch()
    method_after_first = Messages.create

    anthropic_adapter.patch()
    assert Messages.create is method_after_first

    anthropic_adapter.unpatch()


def test_unpatch_restores_original():
    pytest.importorskip("anthropic")
    from tracer.adapters import anthropic_adapter
    try:
        from anthropic.resources.messages.messages import Messages
    except ImportError:
        from anthropic.resources.messages import Messages

    if anthropic_adapter._patched:
        anthropic_adapter.unpatch()

    original = Messages.create
    anthropic_adapter.patch()
    assert Messages.create is not original

    anthropic_adapter.unpatch()
    assert Messages.create is original


def test_unpatch_is_idempotent():
    pytest.importorskip("anthropic")
    from tracer.adapters import anthropic_adapter

    if anthropic_adapter._patched:
        anthropic_adapter.unpatch()

    anthropic_adapter.unpatch()
    anthropic_adapter.unpatch()


# ---------------------------------------------------------------------------
# Streaming warning
# ---------------------------------------------------------------------------

def test_streaming_emits_warning():
    pytest.importorskip("anthropic")
    from tracer.adapters import anthropic_adapter

    stream_sentinel = object()
    original = MagicMock(return_value=stream_sentinel)
    wrapped = anthropic_adapter._wrap_sync(original)
    fake_self = MagicMock()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wrapped(fake_self, model="claude-3-5-haiku-20241022",
                         messages=[], max_tokens=100, stream=True)

    assert result is stream_sentinel
    assert any("streaming" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

def test_parse_extracts_text():
    pytest.importorskip("anthropic")
    from tracer.adapters.anthropic_adapter import _parse_messages_response

    response = _make_response([_make_text_block("Bonjour!")])
    kwargs = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    model, inp, out, params = _parse_messages_response(response, kwargs)

    assert model == "claude-3-5-haiku-20241022"
    assert out["text"] == "Bonjour!"
    assert inp["messages"] == kwargs["messages"]
    assert "tool_calls" not in out


def test_parse_folds_system_prompt_into_messages():
    pytest.importorskip("anthropic")
    from tracer.adapters.anthropic_adapter import _parse_messages_response

    response = _make_response([_make_text_block("Hi")])
    kwargs = {
        "model": "claude-3-5-haiku-20241022",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    _, inp, _, _ = _parse_messages_response(response, kwargs)

    assert inp["messages"][0]["role"] == "system"
    assert inp["messages"][0]["content"] == "You are a helpful assistant."
    assert inp["messages"][1]["role"] == "user"


def test_parse_extracts_tool_calls():
    pytest.importorskip("anthropic")
    from tracer.adapters.anthropic_adapter import _parse_messages_response

    tool_block = _make_tool_use_block(name="get_weather", input_dict={"city": "Paris"})
    response = _make_response([tool_block])
    kwargs = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Weather?"}],
        "max_tokens": 100,
    }
    _, _, out, _ = _parse_messages_response(response, kwargs)

    assert out["text"] == ""
    assert len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    assert tc["id"] == "toolu_1"


def test_parse_captures_tools_in_input():
    pytest.importorskip("anthropic")
    from tracer.adapters.anthropic_adapter import _parse_messages_response

    tools = [{"name": "get_weather", "description": "...", "input_schema": {}}]
    response = _make_response([_make_text_block("ok")])
    kwargs = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "test"}],
        "tools": tools,
        "max_tokens": 100,
    }
    _, inp, _, _ = _parse_messages_response(response, kwargs)

    assert "tools" in inp
    assert inp["tools"][0]["name"] == "get_weather"


def test_parse_extracts_parameters():
    pytest.importorskip("anthropic")
    from tracer.adapters.anthropic_adapter import _parse_messages_response

    response = _make_response([_make_text_block("ok")])
    kwargs = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [],
        "max_tokens": 512,
        "temperature": 0.5,
        "unknown_param": "ignored",
    }
    _, _, _, params = _parse_messages_response(response, kwargs)

    assert params["max_tokens"] == 512
    assert params["temperature"] == 0.5
    assert "unknown_param" not in params


def test_wrap_sync_records_step(tmp_path):
    pytest.importorskip("anthropic")
    from tracer.adapters.anthropic_adapter import _wrap_sync
    from tracer.recorder import init_recorder

    recorder = init_recorder(output_path=str(tmp_path / "test.rpack"))

    fake_response = _make_response([_make_text_block("Paris")])
    original = MagicMock(return_value=fake_response)
    wrapped = _wrap_sync(original)
    fake_self = MagicMock()

    wrapped(
        fake_self,
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=100,
    )

    assert len(recorder.trace.steps) == 1
    assert recorder.trace.steps[0].output["text"] == "Paris"
    recorder._flushed = True
