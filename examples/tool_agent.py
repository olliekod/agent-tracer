"""Multi-step tool-calling agent example.

Demonstrates a two-step OpenAI tool-call loop:
  Step 1 — model decides to call get_weather()
  Step 2 — model reads the tool result and writes a final answer

Run with tracer to capture both steps:

    tracer record examples/tool_agent.py

Requires OPENAI_API_KEY to be set.
The get_weather function is simulated — no external API call is made.
"""

import json

import openai

client = openai.OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. Paris",
                    },
                },
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> str:
    """Simulated weather lookup — returns a fixed string."""
    return f"Sunny, 22°C in {city}"


messages: list[dict] = [
    {"role": "user", "content": "What is the current weather in Paris?"},
]

# Step 1: model inspects the question and decides to call get_weather.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.0,
)

choice = response.choices[0]
assistant_msg = choice.message

# Append assistant message (may contain tool_calls).
messages.append(assistant_msg.model_dump(exclude_unset=False))

if choice.finish_reason == "tool_calls" and assistant_msg.tool_calls:
    for tc in assistant_msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = get_weather(args["city"])
        messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": tc.id,
            }
        )

# Step 2: model reads the tool result and produces a final answer.
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.0,
)

print(response2.choices[0].message.content)
