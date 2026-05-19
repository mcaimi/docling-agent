#!/usr/bin/env python

"""SYSTEM PROMPT FOR THE LLM NLP EXTRACTOR NODE"""

# All 16 config parameters the LLM can extract.
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
]

_SYSTEM_PROMPT = (
    "You are a configuration extraction assistant. Analyze the user's natural language prompt "
    "and extract any relevant configuration parameters. Output ONLY a valid JSON object with "
    "exactly the following keys: {keys}. If a parameter cannot be inferred from the prompt, "
    "set its value to null. Do not include any text, markdown, or explanation outside the JSON."
)


# render prompt
def render() -> str:
    system_message = _SYSTEM_PROMPT.format(
        keys=", ".join(f"`{k}`" for k in _CONFIG_KEYS)
    )

    # return rendered prompt
    return system_message
