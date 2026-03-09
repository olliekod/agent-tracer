# agent-tracer

Crash dumps for AI agents.

Records every LLM call your agent makes and saves everything to a single `.rpack` file on your disk. There are no servers, no background services, no accounts, and no runtime dependencies in the core package.

## Install

```
pip install agent-tracer
```

For OpenAI support:

```
pip install "agent-tracer[openai]"
```

## What is a .rpack file?

A `.rpack` file is a self-contained record of an agent run. Only one file is created per run (however many prompts were involved, it doesn't matter), and it's plain JSON that sits on your disk until you need it.

The `.rpack` file captures everything the model received and output at each step. It stores the full message history, any of the tools and schemas that were in scope, the parameters passed through, the actual response, and any tool calls the model made. It's not a compacted summary or compressed context but instead the actual inputs and outputs of every step.

The idea is that when an agent does something wrong, you should be able to hand someone the `.rpack` file and they can see exactly what happened without re-running anything or needing access to your environment.

`tracer open` reads the `.rpack` and prints it. `tracer replay` steps through it without calling any API. `tracer validate` checks the `.rpack` itself for structural problems.

## Why

Most agent debugging tools give you dashboards and token counts. What you usually need is simpler: what did the model actually see at step 3, and can I reproduce it?

That's what `.rpack` files are for. When a run goes wrong, you open the file, look at the exact prompt the model received, check what tools were available, and see what it decided to do. If you want to step through it again, `tracer replay` does that without touching the API.

## Usage

### Record a run

```
tracer record my_agent.py
```

Runs your script and records every LLM call. Writes a timestamped `.rpack` file when your script exits (e.g. `trace_2026-03-10_21-14-03.rpack`).

Custom output path:

```
tracer record my_agent.py -o my_trace.rpack
```

### Inspect a trace

```
tracer open trace_2026-03-10_21-14-03.rpack
```

Shows every recorded step: model, messages, tool calls, parameters, and response.

```
============================================================
  Run:      a1b2c3d4e5f6
  Provider: openai
  Steps:    2 steps
  Result:   The weather in Paris is sunny at 22°C.
============================================================

--- Step 1 ---
  Model:     gpt-4o-mini
  Time:      2026-03-10 21:14:04 UTC
  Params:    {"temperature": 0.0}
  Input:     1 message(s), 1 tool(s): get_weather
    [user] What is the current weather in Paris?
  Calls:     1
    get_weather({"city": "Paris"})  [call_abc]

--- Step 2 ---
  Model:     gpt-4o-mini
  Time:      2026-03-10 21:14:05 UTC
  Params:    {"temperature": 0.0}
  Input:     3 message(s)
    [user] What is the current weather in Paris?
    [assistant] call get_weather({"city": "Paris"})
    [tool] Sunny, 22°C in Paris
  Output:    The weather in Paris is sunny at 22°C.
```

### Replay a step

```
tracer replay trace_2026-03-10_21-14-03.rpack --step 2
```

Prints the exact input and output for that step. No API call is made.

### Validate an artifact

```
tracer validate trace_2026-03-10_21-14-03.rpack
```

```
OK    trace_2026-03-10_21-14-03.rpack  (2 step(s))
```

## Examples

`examples/simple_agent.py` makes a single OpenAI call.

`examples/tool_agent.py` runs a two-step tool-calling loop where the model calls `get_weather`, reads the result, and writes a final answer.

```
tracer record examples/tool_agent.py
```

## .rpack format

Plain JSON. Open it in any text editor.

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

## How it works

Three layers:

1. **Provider adapters** intercept SDK calls and normalize them into the artifact schema. The OpenAI adapter covers chat completions and the responses API, sync and async. Streaming calls are passed through unrecorded with a warning.

2. **Artifact schema** is provider-agnostic. The `.rpack` format doesn't mirror OpenAI's response shape. Adapters normalize everything before it gets written. Includes a `validate()` method that returns a list of any structural errors.

3. **Replay engine** reads `.rpack` files and returns the recorded outputs. It has no dependency on any provider SDK.

## Development

```
pip install -e ".[dev,openai]"
pytest
```

## License

MIT
