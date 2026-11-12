#!/usr/bin/env python
"""Unit tests for CLI/container logging configuration."""

import json
import logging

from src.utils.logging_setup import JsonFormatter, configure_logging


def _record(msg="hello %s", args=("world",), **extra):
    record = logging.LogRecord("src.test", logging.INFO, __file__, 1, msg, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_merges_static_and_extra_fields():
    formatter = JsonFormatter({"shard_index": 2, "scheduler_run_id": None})
    payload = json.loads(formatter.format(_record(file="a.pdf", stage="convert")))
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "src.test"
    assert payload["shard_index"] == 2
    assert "scheduler_run_id" not in payload  # None static fields are dropped
    assert payload["file"] == "a.pdf"
    assert payload["stage"] == "convert"
    assert "args" not in payload and "msecs" not in payload


def test_configure_logging_json(capsys):
    configure_logging("DEBUG", "json", {"shard_index": 0})
    logging.getLogger("src.graph.test").debug("dbg", extra={"file": "x.md"})
    logging.getLogger("thirdparty").info("noise")  # root stays at WARNING
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines == [
        {**lines[0], "msg": "dbg", "level": "DEBUG", "file": "x.md", "shard_index": 0}
    ]
    configure_logging()  # restore text output for other tests
