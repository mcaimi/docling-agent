#!/usr/bin/env python

import json
from typing import Any

try:
    from openai import OpenAI
    from .prompts.extractor import render as _system_message_render
    from .prompts.extractor import _CONFIG_KEYS, _SENSITIVE_KEYS
except ImportError as e:
    raise ImportError(f"openai package is required: {e}") from e


# extract config keys from natural language user prompt
def extract_config_from_prompt(
    user_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Send user_prompt to an LLM and parse extracted config as JSON.

    Returns a dict with all 16 config keys. Unextracted keys get value None.
    If the LLM response is not valid JSON, returns all None. Keys in
    _SENSITIVE_KEYS (network endpoints, credentials, folder_path) are never
    taken from the model's output -- they always stay None here, regardless
    of what the prompt or the model response contains.
    """
    defaults = {k: None for k in _CONFIG_KEYS}

    if not user_prompt or not user_prompt.strip():
        print("No User Prompt. Using Defaults...")
        return defaults

    client = OpenAI(base_url=base_url, api_key=api_key)

    # build system message
    system_message = _system_message_render()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM Exception {e}")
        return defaults

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return defaults
        result = dict(defaults)
        for key in _CONFIG_KEYS:
            if key in _SENSITIVE_KEYS:
                continue
            value = parsed.get(key)
            if value is not None:
                result[key] = value
        return result
    except (json.JSONDecodeError, ValueError):
        return defaults


# merge state dictionaries
# update state with overrides extracted by the LLM if any
def merge_overrides(state: dict, overrides: dict) -> dict:
    # apply overrides, but never let network endpoints, credentials, or
    # folder_path be set from NLP-extracted (i.e. untrusted free-text) input,
    # even if a caller passes them in `overrides` directly.
    for k in overrides.keys():
        if k in _SENSITIVE_KEYS:
            continue
        if k in state.keys():
            # override
            state[k] = overrides.get(k)

    # return
    return state
