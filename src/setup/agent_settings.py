#!/usr/bin/env python
"""
Resolved agent configuration, driven by ``settings_spec.SETTINGS_SPEC``.

Each setting prefers a non-empty environment value (its ``env`` var, then any
``env_fallbacks``), otherwise the spec default. An optional ``.env`` file is
loaded into ``os.environ`` first; containers pass configuration purely through
the environment and skip it.
"""

import os
from pathlib import Path
from typing import Any, Mapping

# import required modules
try:
    from dotenv import load_dotenv
    from ..utils import _parse_json_object_str
    from .settings_spec import SETTINGS_SPEC, SettingSpec
except ImportError:
    raise ImportError("python-dotenv is required to use this module")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


# loads an env variable and strips it of whitespace
# None if not set
def _strip_or_none(key: str, environ: Mapping[str, str] | None = None) -> str | None:
    raw = (os.environ if environ is None else environ).get(key)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


# raw env value for a setting: primary env var, then fallbacks
def _raw_value(spec: SettingSpec, environ: Mapping[str, str] | None = None) -> str | None:
    for key in (spec.env, *spec.env_fallbacks):
        raw = _strip_or_none(key, environ)
        if raw is not None:
            return raw
    return None


# parse one setting; unparseable int/bool values fall back to the default,
# invalid JSON raises (a malformed header/param object is a config error)
def _resolve_setting(spec: SettingSpec, environ: Mapping[str, str] | None = None) -> Any:
    raw = _raw_value(spec, environ)
    if raw is None:
        return spec.resolve_default()
    if spec.type == "int":
        try:
            return int(raw)
        except ValueError:
            return spec.resolve_default()
    if spec.type == "bool":
        lowered = raw.lower()
        if lowered in _TRUE:
            return 1
        if lowered in _FALSE:
            return 0
        try:
            return int(int(raw) != 0)
        except ValueError:
            return spec.resolve_default()
    if spec.type == "json":
        return _parse_json_object_str(raw)
    return raw


# load the environment variables, optionally from a .env file first
def load_environment_variables(envfile: str | None = None) -> dict[str, Any]:
    if envfile:
        e_file = Path(envfile)
        if not e_file.exists():
            raise FileNotFoundError(f"Environment file {e_file} not found")
        load_dotenv(e_file)  # load the environment variables
    return _get_settings_from_environment()


# get the settings
def _get_settings_from_environment() -> dict[str, Any]:
    return settings_from_mapping(os.environ)


# resolve every setting from an explicit mapping instead of os.environ
# (lets callers resolve settings without mutating os.environ)
def settings_from_mapping(environ: Mapping[str, str]) -> dict[str, Any]:
    return {spec.key: _resolve_setting(spec, environ) for spec in SETTINGS_SPEC}


# export the settings
__all__ = [
    "load_environment_variables",
    "settings_from_mapping",
]
