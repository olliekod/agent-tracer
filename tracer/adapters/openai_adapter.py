"""OpenAI SDK adapter — intercepts chat completion and responses API calls.

Patches four entry points:
  - client.chat.completions.create()       (sync)
  - client.chat.completions.create()       (async)
  - client.responses.create()              (sync,  SDK v1.66+)
  - client.responses.create()              (async, SDK v1.66+)

Streaming calls are passed through unrecorded and emit a warning.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

from tracer.recorder import get_recorder

_PARAM_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_output_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "instructions",
)

# Stored originals for unpatch() and idempotency guard.
_originals: dict[str, Any] = {}
_patched: bool = False


def _extract_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: kwargs[k] for k in _PARAM_KEYS if k in kwargs}


def _record(model: str, input_data: dict, output_data: dict, params: dict) -> None:
    get_recorder().record_step(
        model=model,
        input_data=input_data,
        output_data=output_data,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Chat Completions API
# ---------------------------------------------------------------------------

def _parse_chat_completion(response: Any, kwargs: dict[str, Any]) -> tuple:
    """Normalize a ChatCompletion response into schema-ready dicts."""
    model = kwargs.get("model", getattr(response, "model", "unknown"))
    messages = list(kwargs.get("messages", []))
    params = _extract_parameters(kwargs)

    text = ""
    tool_calls: list[dict[str, Any]] = []

    if hasattr(response, "choices") and response.choices:
        msg = getattr(response.choices[0], "message", None)
        if msg:
            text = msg.content or ""
            if msg.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

    output: dict[str, Any] = {"text": text}
    if tool_calls:
        output["tool_calls"] = tool_calls

    inp: dict[str, Any] = {"messages": messages}
    if "tools" in kwargs:
        inp["tools"] = list(kwargs["tools"])
    if "tool_choice" in kwargs:
        inp["tool_choice"] = kwargs["tool_choice"]

    return model, inp, output, params


def _wrap_chat_sync(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        if kwargs.get("stream"):
            warnings.warn(
                "tracer: streaming calls are not recorded in v0.1. "
                "Pass stream=False to capture this step.",
                stacklevel=2,
            )
            return response
        model, inp, out, params = _parse_chat_completion(response, kwargs)
        _record(model, inp, out, params)
        return response
    return wrapper


def _wrap_chat_async(original):
    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        response = await original(self, *args, **kwargs)
        if kwargs.get("stream"):
            warnings.warn(
                "tracer: streaming calls are not recorded in v0.1. "
                "Pass stream=False to capture this step.",
                stacklevel=2,
            )
            return response
        model, inp, out, params = _parse_chat_completion(response, kwargs)
        _record(model, inp, out, params)
        return response
    return wrapper


# ---------------------------------------------------------------------------
# Responses API  (SDK v1.66+)
# ---------------------------------------------------------------------------

def _parse_responses_api(response: Any, kwargs: dict[str, Any]) -> tuple:
    """Normalize a Responses API response into schema-ready dicts."""
    model = kwargs.get("model", getattr(response, "model", "unknown"))

    raw_input = kwargs.get("input", "")
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    else:
        messages = list(raw_input)

    params = _extract_parameters(kwargs)

    text = getattr(response, "output_text", "") or ""
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "function_call":
            tool_calls.append({
                "id": getattr(item, "call_id", ""),
                "type": "function",
                "function": {
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", ""),
                },
            })

    output: dict[str, Any] = {"text": text}
    if tool_calls:
        output["tool_calls"] = tool_calls

    inp: dict[str, Any] = {"messages": messages}
    if "tools" in kwargs:
        inp["tools"] = list(kwargs["tools"])

    return model, inp, output, params


def _wrap_responses_sync(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        if kwargs.get("stream"):
            warnings.warn(
                "tracer: streaming calls are not recorded in v0.1. "
                "Pass stream=False to capture this step.",
                stacklevel=2,
            )
            return response
        model, inp, out, params = _parse_responses_api(response, kwargs)
        _record(model, inp, out, params)
        return response
    return wrapper


def _wrap_responses_async(original):
    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        response = await original(self, *args, **kwargs)
        if kwargs.get("stream"):
            warnings.warn(
                "tracer: streaming calls are not recorded in v0.1. "
                "Pass stream=False to capture this step.",
                stacklevel=2,
            )
            return response
        model, inp, out, params = _parse_responses_api(response, kwargs)
        _record(model, inp, out, params)
        return response
    return wrapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch() -> None:
    """Monkey-patch the OpenAI SDK so all LLM calls are recorded.

    Idempotent — calling patch() more than once has no effect.
    Call unpatch() to restore the original methods.
    """
    global _patched
    if _patched:
        return

    try:
        import openai  # noqa: F401
    except ImportError:
        raise ImportError(
            "openai>=1.0 is required for the OpenAI adapter. "
            "Install it with:  pip install openai"
        ) from None

    from openai.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    _originals["chat_sync"] = Completions.create
    _originals["chat_async"] = AsyncCompletions.create
    Completions.create = _wrap_chat_sync(Completions.create)  # type: ignore[assignment]
    AsyncCompletions.create = _wrap_chat_async(AsyncCompletions.create)  # type: ignore[assignment]

    # Responses API — only present in newer SDK versions.
    try:
        from openai.resources.responses.responses import (
            AsyncResponses,
            Responses,
        )
    except ImportError:
        try:
            from openai.resources.responses import (  # type: ignore[no-redef]
                AsyncResponses,
                Responses,
            )
        except (ImportError, AttributeError):
            _patched = True
            return

    _originals["responses_sync"] = Responses.create
    _originals["responses_async"] = AsyncResponses.create
    Responses.create = _wrap_responses_sync(Responses.create)  # type: ignore[assignment]
    AsyncResponses.create = _wrap_responses_async(AsyncResponses.create)  # type: ignore[assignment]

    _patched = True


def unpatch() -> None:
    """Restore the original OpenAI SDK methods.

    Idempotent — calling unpatch() when not patched has no effect.
    """
    global _patched
    if not _patched:
        return

    from openai.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    if "chat_sync" in _originals:
        Completions.create = _originals.pop("chat_sync")  # type: ignore[assignment]
    if "chat_async" in _originals:
        AsyncCompletions.create = _originals.pop("chat_async")  # type: ignore[assignment]

    try:
        from openai.resources.responses.responses import (
            AsyncResponses,
            Responses,
        )
    except ImportError:
        try:
            from openai.resources.responses import (  # type: ignore[no-redef]
                AsyncResponses,
                Responses,
            )
        except (ImportError, AttributeError):
            _patched = False
            return

    if "responses_sync" in _originals:
        Responses.create = _originals.pop("responses_sync")  # type: ignore[assignment]
    if "responses_async" in _originals:
        AsyncResponses.create = _originals.pop("responses_async")  # type: ignore[assignment]

    _patched = False
