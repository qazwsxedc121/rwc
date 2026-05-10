"""Stateless LLM client — each call is a fresh API request with no history."""
import json
import os
import re
from typing import Any

import anthropic


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY."
        )
    kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def chat(
    user_message: str,
    *,
    system: str = "",
    model: str = "deepseek-v4-flash",
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> str:
    """Single stateless chat. No conversation history, no caching."""
    client = _get_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
        "thinking": {"type": "disabled"},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    for block in response.content:
        if block.type == "text":
            return block.text
    raise RuntimeError(f"No text block in response: {response.content}")


def _extract_json(text: str) -> Any:
    """Extract and parse JSON from LLM output, handling markdown fences and
    trailing commentary."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()

    # Try parsing whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try stripping trailing chars (LLM often appends "." or commentary)
    for end in range(len(text), max(0, len(text) - 50), -1):
        try:
            return json.loads(text[:end])
        except json.JSONDecodeError:
            continue

    # Brace/bracket matching as last resort
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        start = text.find(open_c)
        if start == -1:
            continue
        depth = 0
        in_string = False
        for i in range(start, len(text)):
            c = text[i]
            if c == '"' and (i == start or text[i - 1] != '\\\\'):
                in_string = not in_string
            if in_string:
                continue
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def chat_json(
    user_message: str,
    *,
    system: str = "",
    model: str = "deepseek-v4-flash",
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> Any:
    """Single stateless chat, parse response as JSON."""
    text = chat(user_message, system=system, model=model, max_tokens=max_tokens, temperature=temperature)
    return _extract_json(text)
