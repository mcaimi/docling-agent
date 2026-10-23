#!/usr/bin/env python

import json
from typing import Any

try:
    from openai import OpenAI
    from .prompts.extractor import render as _system_message_render
    from .prompts.extractor import _CONFIG_KEYS, _SENSITIVE_KEYS
except ImportError as e:
    raise ImportError(f"openai package is required: {e}") from e

# Keys with a fixed, known-good set of values. The extraction LLM is free-text
# driven (untrusted input in, structured output expected), so its output is
# validated here rather than trusted outright.
_ENUM_OVERRIDES: dict[str, frozenset[str]] = {
    "chunking_strategy": frozenset({"fixed", "hybrid"}),
    "conversion_pipeline": frozenset({"standard", "vlm"}),
    "vector_store_backend": frozenset({"none", "chroma", "milvus"}),
}

# Keys that must be a positive integer (chunk_overlap may be zero).
_POSITIVE_INT_OVERRIDES = frozenset({"chunk_size", "embedding_dimensions"})
_NON_NEGATIVE_INT_OVERRIDES = frozenset({"chunk_overlap"})


def _is_valid_override(key: str, current: Any, value: Any) -> bool:
    """Sanity-check one NLP-extracted override before it's allowed to merge into
    agent settings.

    The model is asked to return specific types per key, but nothing stops a
    model from hallucinating a wrong type or an out-of-range value (e.g. a
    negative chunk_size, or ``chunking_strategy: "semantic"`` for a strategy
    that doesn't exist). Downstream code degrades gracefully for most of these
    today, but rejecting bad values at the merge boundary means safety doesn't
    depend on every call site downstream getting that handling right.
    """
    allowed = _ENUM_OVERRIDES.get(key)
    if allowed is not None:
        return isinstance(value, str) and value.strip().lower() in allowed

    if key in _POSITIVE_INT_OVERRIDES:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0

    if key in _NON_NEGATIVE_INT_OVERRIDES:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0

    # Generic fallback: an override must be the same broad type as the setting
    # it would replace (e.g. a string can't override a string, but not a dict).
    if current is None:
        return True
    if isinstance(current, bool):
        return isinstance(value, bool)
    if isinstance(current, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(current, str):
        return isinstance(value, str) and bool(value.strip())
    return isinstance(value, type(current))


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
    # even if a caller passes them in `overrides` directly. Remaining keys are
    # still sanity-checked against their expected type/range/enum before being
    # applied -- see _is_valid_override.
    for k, value in overrides.items():
        if k in _SENSITIVE_KEYS:
            continue
        if k not in state:
            continue
        if not _is_valid_override(k, state.get(k), value):
            print(f"Ignoring invalid NLP-extracted override for '{k}': {value!r}")
            continue
        # enum values are validated case-insensitively; normalize before storing
        # so downstream exact-match comparisons (e.g. `strategy == "hybrid"`) hold
        if k in _ENUM_OVERRIDES:
            value = value.strip().lower()
        state[k] = value

    # return
    return state
