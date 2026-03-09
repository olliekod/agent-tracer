"""Tests for the OpenAI adapter: patch/unpatch idempotency, streaming warning,
and response normalization. Uses unittest.mock to avoid real API calls."""

import importlib
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch as mock_patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake OpenAI response objects
# ---------------------------------------------------------------------------

def _make_chat_response(content="Hello!", tool_calls=None, model="gpt-4"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model=model)


def _make_tool_call(name="get_weather", arguments='{"city": "Paris"}', tc_id="call_1"):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=tc_id, type="function", function=fn)


# ---------------------------------------------------------------------------
# Patch/unpatch idempotency
# ---------------------------------------------------------------------------

def test_patch_is_idempotent():
    """Calling patch() twice should not double-wrap the SDK methods."""
    pytest.importorskip("openai")
    from tracer.adapters import openai_adapter
    from openai.resources.chat.completions import Completions

    # Reset state in case another test already patched.
    if openai_adapter._patched:
        openai_adapter.unpatch()

    openai_adapter.patch()
    method_after_first_patch = Completions.create

    openai_adapter.patch()  # second call — should be a no-op
    assert Completions.create is method_after_first_patch

    openai_adapter.unpatch()


def test_unpatch_restores_original():
    """unpatch() should restore the method that existed before patch()."""
    pytest.importorskip("openai")
    from tracer.adapters import openai_adapter
    from openai.resources.chat.completions import Completions

    if openai_adapter._patched:
        openai_adapter.unpatch()

    original = Completions.create
    openai_adapter.patch()
    assert Completions.create is not original

    openai_adapter.unpatch()
    assert Completions.create is original


def test_unpatch_is_idempotent():
    """Calling unpatch() when not patched should not raise."""
    pytest.importorskip("openai")
    from tracer.adapters import openai_adapter

    if openai_adapter._patched:
        openai_adapter.unpatch()

    # Should be a no-op and not raise.
    openai_adapter.unpatch()
    openai_adapter.unpatch()


# ---------------------------------------------------------------------------
# Streaming warning
# ---------------------------------------------------------------------------

def test_streaming_emits_warning():
    """A stream=True call should pass through and emit a UserWarning."""
    pytest.importorskip("openai")
    from tracer.adapters import openai_adapter

    if openai_adapter._patched:
        openai_adapter.unpatch()

    # Fake streamed response sentinel.
    stream_sentinel = object()

    original_create = MagicMock(return_value=stream_sentinel)

    wrapped = openai_adapter._wrap_chat_sync(original_create)
    fake_self = MagicMock()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wrapped(fake_self, model="gpt-4", messages=[], stream=True)

    assert result is stream_sentinel
    assert any("streaming" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

def test_parse_chat_completion_extracts_text():
    pytest.importorskip("openai")
    from tracer.adapters.openai_adapter import _parse_chat_completion

    response = _make_chat_response(content="Bonjour!")
    kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    model, inp, out, params = _parse_chat_completion(response, kwargs)

    assert model == "gpt-4"
    assert out["text"] == "Bonjour!"
    assert inp["messages"] == kwargs["messages"]
    assert "tool_calls" not in out


def test_parse_chat_completion_extracts_tool_calls():
    pytest.importorskip("openai")
    from tracer.adapters.openai_adapter import _parse_chat_completion

    tc = _make_tool_call()
    response = _make_chat_response(content=None, tool_calls=[tc])
    kwargs = {"model": "gpt-4", "messages": []}
    _, _, out, _ = _parse_chat_completion(response, kwargs)

    assert out["text"] == ""
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["function"]["name"] == "get_weather"
    assert out["tool_calls"][0]["id"] == "call_1"


def test_parse_chat_completion_extracts_parameters():
    pytest.importorskip("openai")
    from tracer.adapters.openai_adapter import _parse_chat_completion

    response = _make_chat_response()
    kwargs = {
        "model": "gpt-4",
        "messages": [],
        "temperature": 0.2,
        "max_tokens": 100,
        "unknown_param": "ignored",
    }
    _, _, _, params = _parse_chat_completion(response, kwargs)

    assert params["temperature"] == 0.2
    assert params["max_tokens"] == 100
    assert "unknown_param" not in params


def test_wrap_chat_sync_records_step(tmp_path):
    """End-to-end: wrapped sync call records a step via the global recorder."""
    pytest.importorskip("openai")
    from tracer.adapters.openai_adapter import _wrap_chat_sync
    from tracer.recorder import init_recorder

    recorder = init_recorder(output_path=str(tmp_path / "test.rpack"))

    fake_response = _make_chat_response(content="Paris")
    original = MagicMock(return_value=fake_response)
    wrapped = _wrap_chat_sync(original)
    fake_self = MagicMock()

    wrapped(fake_self, model="gpt-4", messages=[{"role": "user", "content": "Capital?"}])

    assert len(recorder.trace.steps) == 1
    assert recorder.trace.steps[0].output["text"] == "Paris"
    recorder._flushed = True  # prevent atexit write


def test_parse_chat_completion_captures_tools():
    """Tool schemas and tool_choice sent to the model must be preserved in input."""
    pytest.importorskip("openai")
    from tracer.adapters.openai_adapter import _parse_chat_completion

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    response = _make_chat_response(content=None, tool_calls=[_make_tool_call()])
    kwargs = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": tools,
        "tool_choice": "auto",
    }
    _, inp, _, _ = _parse_chat_completion(response, kwargs)

    assert "tools" in inp
    assert len(inp["tools"]) == 1
    assert inp["tools"][0]["function"]["name"] == "get_weather"
    assert inp["tool_choice"] == "auto"


def test_parse_chat_completion_no_tools_when_absent():
    """When no tools are passed, input must not contain a 'tools' key."""
    pytest.importorskip("openai")
    from tracer.adapters.openai_adapter import _parse_chat_completion

    response = _make_chat_response(content="Bonjour!")
    kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    _, inp, _, _ = _parse_chat_completion(response, kwargs)

    assert "tools" not in inp
    assert "tool_choice" not in inp
