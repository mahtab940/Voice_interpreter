
import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY missing in .env")

MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-sonnet-5-5"
)

client = Anthropic(
    api_key=API_KEY,
    timeout=30.0,
    max_retries=1
)

SYSTEM_PROMPT = """
You are a friendly AI voice tutor.

Your responsibilities:
- Explain difficult topics in simple English.
- Answer questions clearly and accurately.
- Keep answers short and conversational.
- Help students understand concepts step by step.
- Remember the conversation context.
- Avoid unnecessary formatting.
"""

history = []


def ask_ai(question):
    global history

    start = time.perf_counter()

    messages = history + [
        {
            "role": "user",
            "content": question
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    answer = " ".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()

    if not answer:
        raise RuntimeError("Claude returned no text")

    history.extend([
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": answer
        }
    ])

    latency = time.perf_counter() - start

    return answer, latency
