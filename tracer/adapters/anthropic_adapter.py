"""Anthropic SDK adapter — intercepts messages.create calls.

Patches two entry points:
  - client.messages.create()       (sync)
  - client.messages.create()       (async)

The Anthropic API takes the system prompt as a separate top-level parameter.
This adapter normalizes it into the messages list as a {"role": "system", ...}
entry so the artifact matches the provider-agnostic schema.

Streaming calls are passed through unrecorded and emit a warning.
"""

from __future__ import annotations

import functools
import json
import warnings
from typing import Any

from tracer.recorder import get_recorder

_PARAM_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
)

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


def _parse_messages_response(response: Any, kwargs: dict[str, Any]) -> tuple:
    """Normalize a Messages API response into schema-ready dicts."""
    model = kwargs.get("model", getattr(response, "model", "unknown"))

    messages = list(kwargs.get("messages", []))

    # Anthropic takes the system prompt as a separate top-level kwarg.
    # Fold it into the messages list so the artifact is self-contained.
    system = kwargs.get("system")
    if system:
        messages = [{"role": "system", "content": system}] + messages

    params = _extract_parameters(kwargs)

    text = ""
    tool_calls: list[dict[str, Any]] = []

    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "") or ""
        elif block_type == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {})),
                },
            })

    output: dict[str, Any] = {"text": text}
    if tool_calls:
        output["tool_calls"] = tool_calls

    inp: dict[str, Any] = {"messages": messages}
    if "tools" in kwargs:
        inp["tools"] = list(kwargs["tools"])
    if "tool_choice" in kwargs:
        inp["tool_choice"] = kwargs["tool_choice"]

    return model, inp, output, params


def _wrap_sync(original):
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
        model, inp, out, params = _parse_messages_response(response, kwargs)
        _record(model, inp, out, params)
        return response
    return wrapper


def _wrap_async(original):
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
        model, inp, out, params = _parse_messages_response(response, kwargs)
        _record(model, inp, out, params)
        return response
    return wrapper


def patch() -> None:
    """Monkey-patch the Anthropic SDK so all Messages calls are recorded.

    Idempotent — calling patch() more than once has no effect.
    Call unpatch() to restore the original methods.
    """
    global _patched
    if _patched:
        return

    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise ImportError(
            "anthropic>=0.18 is required for the Anthropic adapter. "
            "Install it with:  pip install anthropic"
        ) from None

    try:
        from anthropic.resources.messages.messages import AsyncMessages, Messages
    except ImportError:
        from anthropic.resources.messages import AsyncMessages, Messages  # type: ignore[no-redef]

    _originals["sync"] = Messages.create
    _originals["async"] = AsyncMessages.create
    Messages.create = _wrap_sync(Messages.create)  # type: ignore[assignment]
    AsyncMessages.create = _wrap_async(AsyncMessages.create)  # type: ignore[assignment]

    _patched = True


def unpatch() -> None:
    """Restore the original Anthropic SDK methods.

    Idempotent — calling unpatch() when not patched has no effect.
    """
    global _patched
    if not _patched:
        return

    try:
        from anthropic.resources.messages.messages import AsyncMessages, Messages
    except ImportError:
        from anthropic.resources.messages import AsyncMessages, Messages  # type: ignore[no-redef]

    if "sync" in _originals:
        Messages.create = _originals.pop("sync")  # type: ignore[assignment]
    if "async" in _originals:
        AsyncMessages.create = _originals.pop("async")  # type: ignore[assignment]

    _patched = False
