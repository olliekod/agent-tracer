"""CLI entry point for agent-tracer."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

_SUPPORTED_PROVIDERS = ("openai", "anthropic")
_CONTENT_TRUNCATE = 200  # chars before truncating message content in output


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_record(args: argparse.Namespace) -> None:
    """Run a Python script while recording all LLM calls."""
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"Error: {args.script} not found", file=sys.stderr)
        sys.exit(1)

    provider = args.provider
    patch_fn = _load_adapter(provider)

    from tracer.recorder import init_recorder

    recorder = init_recorder(output_path=args.output, provider=provider)
    patch_fn()

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    sys.argv = [str(script_path)]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit:
        pass
    finally:
        recorder.flush()
        print(f"Trace saved to {recorder.output_path}")


def cmd_open(args: argparse.Namespace) -> None:
    """Print all steps in an .rpack file."""
    from tracer.replay import load_trace

    try:
        trace = load_trace(args.rpack)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    step_word = "step" if len(trace.steps) == 1 else "steps"
    print(f"{'=' * 60}")
    print(f"  Run:      {trace.run_id}")
    print(f"  Provider: {trace.provider}")
    print(f"  Steps:    {len(trace.steps)} {step_word}")
    if trace.final_output:
        print(f"  Result:   {_truncate(trace.final_output)}")
    print(f"{'=' * 60}")
    print()

    for step in trace.steps:
        _print_step(
            step.step, step.model, step.timestamp,
            step.parameters, step.input, step.output,
        )


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a single step from an .rpack file."""
    from tracer.replay import load_trace, replay_step

    try:
        trace = load_trace(args.rpack)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        result = replay_step(trace, args.step)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_step(
        result["step"], result["model"], result["timestamp"],
        result["parameters"], result["input"], result["output"],
    )


def cmd_validate(args: argparse.Namespace) -> None:
    """Check an .rpack file for schema integrity."""
    from tracer.replay import load_trace

    try:
        trace = load_trace(args.rpack)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    errors = trace.validate()
    if errors:
        print(f"FAIL  {args.rpack}  ({len(errors)} error(s))")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"OK    {args.rpack}  ({len(trace.steps)} step(s))")


# ---------------------------------------------------------------------------
# Adapter loader
# ---------------------------------------------------------------------------

def _load_adapter(provider: str):
    """Return the patch() function for the given provider."""
    if provider == "openai":
        from tracer.adapters.openai_adapter import patch
        return patch
    if provider == "anthropic":
        from tracer.adapters.anthropic_adapter import patch
        return patch
    supported = ", ".join(_SUPPORTED_PROVIDERS)
    print(
        f"Error: unknown provider {provider!r}. Supported: {supported}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = _CONTENT_TRUNCATE) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(text) - limit} more chars]"


def _fmt_timestamp(ts: str) -> str:
    """Return a compact local-time string from an ISO timestamp, or the raw value."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        dt_local = dt.astimezone()
        return dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return ts


def _print_message(msg: dict) -> None:
    """Print a single message from the conversation history."""
    role = msg.get("role", "?")
    content = msg.get("content") or ""

    if content:
        print(f"  [{role}] {_truncate(str(content))}")
    elif msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            args_str = _truncate(fn.get("arguments", ""), 80)
            print(f"  [{role}] call {fn.get('name', '?')}({args_str})")
    else:
        print(f"  [{role}]")


def _print_step(
    step_num: int,
    model: str,
    timestamp: str,
    parameters: dict,
    input_data: dict,
    output_data: dict,
) -> None:
    print(f"--- Step {step_num} ---")
    print(f"  Model:     {model}")
    if timestamp:
        print(f"  Time:      {_fmt_timestamp(timestamp)}")
    if parameters:
        print(f"  Params:    {json.dumps(parameters)}")

    messages = input_data.get("messages", [])
    tools = input_data.get("tools", [])
    tool_choice = input_data.get("tool_choice")

    if messages:
        suffix = ""
        if tools:
            tool_names = [
                t.get("function", {}).get("name", "?")
                for t in tools
                if t.get("type") == "function"
            ]
            suffix = f", {len(tools)} tool(s): {', '.join(tool_names)}"
        if tool_choice and tool_choice != "auto":
            suffix += f", tool_choice={tool_choice!r}"
        print(f"  Input:     {len(messages)} message(s){suffix}")
        for msg in messages:
            _print_message(msg)

    output_text = output_data.get("text", "")
    tool_calls = output_data.get("tool_calls", [])

    if output_text:
        print(f"  Output:    {_truncate(output_text)}")
    if tool_calls:
        print(f"  Calls:     {len(tool_calls)}")
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_id = tc.get("id", "")
            args_str = _truncate(fn.get("arguments", ""), 80)
            print(f"    {fn.get('name', '?')}({args_str})  [{tc_id}]")
    if not output_text and not tool_calls:
        print("  Output:    (empty)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tracer",
        description="Replay debugger for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="Record an agent run")
    rec.add_argument("script", help="Python script to run")
    rec.add_argument("-o", "--output", help="Output .rpack path (default: timestamped)")
    rec.add_argument(
        "--provider",
        default="openai",
        choices=_SUPPORTED_PROVIDERS,
        help="Provider adapter to use (default: openai)",
    )

    opn = sub.add_parser("open", help="Inspect an .rpack file")
    opn.add_argument("rpack", help="Path to .rpack file")

    rep = sub.add_parser("replay", help="Replay a specific step")
    rep.add_argument("rpack", help="Path to .rpack file")
    rep.add_argument("--step", type=int, required=True, help="Step number to replay")

    val = sub.add_parser("validate", help="Check .rpack schema integrity")
    val.add_argument("rpack", help="Path to .rpack file")

    args = parser.parse_args()

    handlers = {
        "record": cmd_record,
        "open": cmd_open,
        "replay": cmd_replay,
        "validate": cmd_validate,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
