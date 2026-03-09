"""Minimal agent example — makes one OpenAI call.

Usage:
    tracer record examples/simple_agent.py

Requires OPENAI_API_KEY in the environment.
"""

import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.0,
)

print(response.choices[0].message.content)
