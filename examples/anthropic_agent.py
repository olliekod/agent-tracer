"""Two-step Anthropic tool-calling example.

The model decides to call get_weather(), reads the result, and writes a
final answer. Both steps are recorded by Tracer.

Run with:
    tracer record examples/anthropic_agent.py --provider anthropic

Requires ANTHROPIC_API_KEY to be set.
The get_weather function is simulated — no external API call is made.
"""

import json

import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Return current weather conditions for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. Paris",
                },
            },
            "required": ["city"],
        },
    }
]


def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"


messages = [
    {"role": "user", "content": "What is the current weather in Paris?"},
]

# Step 1: model decides to call get_weather.
response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    messages=messages,
    tools=tools,
    temperature=0.0,
)

# Append the assistant turn and any tool results.
messages.append({"role": "assistant", "content": response.content})

for block in response.content:
    if block.type == "tool_use":
        result = get_weather(block.input["city"])
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            ],
        })

# Step 2: model reads the tool result and writes the final answer.
response2 = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    messages=messages,
    temperature=0.0,
)

print(response2.content[0].text)
