#!/usr/bin/env python
"""The committed settings contract must match the code.

Launchers (docling-scheduler) vendor ``settings-schema.json`` instead of
importing the agent; regenerate it with
``python -m src.docling_agent --print-settings-schema > settings-schema.json``.
"""

import json
from pathlib import Path

from src.graph.document_paths import DOCUMENT_GLOBS
from src.setup.settings_spec import settings_contract

CONTRACT_FILE = Path(__file__).resolve().parents[2] / "settings-schema.json"


def test_committed_contract_is_up_to_date():
    committed = json.loads(CONTRACT_FILE.read_text())
    assert committed == settings_contract(DOCUMENT_GLOBS), (
        "settings-schema.json is stale: regenerate it with "
        "`python -m src.docling_agent --print-settings-schema > settings-schema.json`"
    )
