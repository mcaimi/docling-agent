#!/usr/bin/env python

"""SYSTEM PROMPT FOR THE LLM NLP EXTRACTOR NODE"""

# All 16 config parameters the extraction contract exposes. Values for keys in
# _SENSITIVE_KEYS are never actually taken from the LLM's output (see
# prompt_extraction.extract_config_from_prompt) -- they stay listed here only so
# the returned dict always has the documented 16 keys, each defaulting to None.
_CONFIG_KEYS = [
    "folder_path",
    "embedding_api_key",
    "embedding_base_url",
    "embedding_model",
    "embedding_dimensions",
    "conversion_pipeline",
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
    "llm_model",
    "base_url",
    "api_key",
    "vlm_api_url",
    "vlm_api_headers",
    "vlm_preset",
    "vector_store_backend",
    "vector_store_collection",
]

# Keys that control network destinations, credentials, or filesystem access.
# Free-text user input must never be able to set these -- a crafted prompt
# could otherwise redirect embedding/VLM traffic to an attacker endpoint,
# inject outbound headers, or read an arbitrary directory. These are excluded
# from the extraction prompt itself and, as a second line of defense, filtered
# out again in extract_config_from_prompt/merge_overrides even if a model
# response includes them.
_SENSITIVE_KEYS = frozenset(
    {
        "folder_path",
        "embedding_api_key",
        "embedding_base_url",
        "api_key",
        "base_url",
        "vlm_api_url",
        "vlm_api_headers",
    }
)

# Keys the LLM is actually asked to extract from free text.
_EXTRACTABLE_KEYS = [k for k in _CONFIG_KEYS if k not in _SENSITIVE_KEYS]

_SYSTEM_PROMPT = (
    "You are a configuration extraction assistant. Analyze the user's natural language prompt "
    "and extract any relevant configuration parameters. Output ONLY a valid JSON object with "
    "exactly the following keys: {keys}. If a parameter cannot be inferred from the prompt, "
    "set its value to null. Do not include any text, markdown, or explanation outside the JSON."
)


# render prompt
def render() -> str:
    system_message = _SYSTEM_PROMPT.format(
        keys=", ".join(f"`{k}`" for k in _EXTRACTABLE_KEYS)
    )

    # return rendered prompt
    return system_message
