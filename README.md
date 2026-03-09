# agent-tracer

Crash dumps for AI agents.

Records AI agent runs and enables deterministic replay of every LLM interaction.
No servers, no databases — just a single `.rpack` artifact file.

## Install

```
pip install agent-tracer
```

For OpenAI support:

```
pip install "agent-tracer[openai]"
```

## Quickstart

### Record a run

```
tracer record my_agent.py
```

Runs your script while intercepting all LLM calls. On completion, writes a
timestamped `.rpack` file (e.g. `trace_2026-03-10_21-14-03.rpack`).

Specify a custom output path:

```
tracer record my_agent.py -o my_trace.rpack
```

### Inspect the trace

```
tracer open trace_2026-03-10_21-14-03.rpack
```

Prints a summary header and every recorded step: model, messages, tool calls, and response.
Long content is truncated for readability.

```
============================================================
Run:      a1b2c3d4e5f6
Provider: openai
Steps:    2
Result:   The weather in Paris is sunny at 22°C.
============================================================

--- Step 1 ---
Model:     gpt-4o-mini
Timestamp: 2026-03-10T21:14:04+00:00
Params:    {"temperature": 0.0}
Input:     (1 message(s))
  [user] What is the current weather in Paris?
Tool calls: (1)
  get_weather({"city": "Paris"})  [call_abc]

--- Step 2 ---
Model:     gpt-4o-mini
Timestamp: 2026-03-10T21:14:05+00:00
Params:    {"temperature": 0.0}
Input:     (3 message(s))
  [user] What is the current weather in Paris?
  [assistant] call get_weather({"city": "Paris"})
  [tool] Sunny, 22°C in Paris
Output:    The weather in Paris is sunny at 22°C.
```

### Replay a step

```
tracer replay trace_2026-03-10_21-14-03.rpack --step 2
```

Displays the exact input and output for step 2. No API call is made.

### Validate an artifact

```
tracer validate trace_2026-03-10_21-14-03.rpack
```

Checks schema integrity and reports any structural problems.

```
OK    trace_2026-03-10_21-14-03.rpack  (2 step(s))
```

## Examples

`examples/simple_agent.py` — single-turn question, no tools.

`examples/tool_agent.py` — two-step tool-calling loop: model calls `get_weather`,
reads the result, and produces a final answer. Run with:

```
tracer record examples/tool_agent.py
```

## Sample .rpack

```json
{
  "run_id": "a1b2c3d4e5f6",
  "provider": "openai",
  "steps": [
    {
      "step": 1,
      "model": "gpt-4o-mini",
      "input": {
        "messages": [
          {"role": "user", "content": "What is the capital of France?"}
        ]
      },
      "parameters": {
        "temperature": 0.0
      },
      "output": {
        "text": "The capital of France is Paris."
      },
      "timestamp": "2026-03-10T21:14:04+00:00"
    }
  ],
  "final_output": "The capital of France is Paris."
}
```

## Architecture

Three strict layers:

1. **Provider Adapters** — intercept SDK calls and normalize to the generic schema.
   Covers chat completions and the responses API, sync and async.
   Streaming calls are passed through with a warning (not recorded in v0.1).
   `patch()` is idempotent; `unpatch()` restores the originals.

2. **Artifact Schema** — provider-agnostic `.rpack` format (JSON). Stores the full
   message history, tool calls, parameters, and timestamps. Includes a `validate()`
   method that returns a list of structural errors.

3. **Replay Engine** — reads artifacts and simulates steps deterministically.
   Never imports or depends on any provider SDK.

## Development

```
pip install -e ".[dev,openai]"
pytest
```

## License

MIT
