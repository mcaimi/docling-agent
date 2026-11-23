#!/usr/bin/env python
"""Unit tests for keeping bulky payloads (vectors, documents, long text) out of MLflow traces."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.docling_agent as agent_mod
import src.graph.embeddings as embeddings_mod
import src.graph.prompt_extraction as prompt_mod
from src.utils import mlflow_tracing
from src.utils.mlflow_tracing import payload_guard, summarize_payload


class FakeSpan:
    """Minimal stand-in for mlflow.entities.LiveSpan."""

    def __init__(self, inputs=None, outputs=None, attributes=None):
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = dict(attributes or {})

    def set_inputs(self, value):
        self.inputs = value

    def set_outputs(self, value):
        self.outputs = value

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestSummarizePayload:
    def test_scalars_unchanged(self):
        assert summarize_payload({"a": 1, "b": 2.5, "c": True, "d": None, "e": "x"}) == {
            "a": 1, "b": 2.5, "c": True, "d": None, "e": "x"
        }

    def test_vector_is_summarized(self):
        assert summarize_payload([0.1] * 1536) == "<vector dim=1536>"

    def test_list_of_vectors_is_summarized(self):
        assert summarize_payload([[0.1] * 768] * 40) == "<40 vectors dim=768>"

    def test_short_number_list_kept(self):
        assert summarize_payload([1, 2, 3]) == [1, 2, 3]

    def test_long_string_truncated_to_preview(self):
        out = summarize_payload("x" * 5000)
        assert out.startswith("x" * 200)
        assert out.endswith("<truncated, 5000 chars>")
        assert len(out) < 300

    def test_bulky_keys_summarized(self):
        out = summarize_payload(
            {
                "embeddings": [[0.1] * 1536, None, [0.2] * 1536],
                "docling_document": object(),
                "content": "z" * 1000,
                "metadata": {"chunk_index": 3},
            }
        )
        assert out["embeddings"] == "<3 vectors dim=1536>"
        assert out["docling_document"] == "<object>"
        assert out["content"].endswith("<truncated, 1000 chars>")
        assert out["metadata"] == {"chunk_index": 3}

    def test_nested_chunk_records(self):
        records = [{"content": "c" * 500, "metadata": {"file_path": "a.pdf"}}] * 50
        out = summarize_payload({"chunks": records, "doc": {"chunks_kept": records[:2]}})
        assert out["chunks"] == "<50 items>"
        assert out["doc"]["chunks_kept"][0]["content"].endswith("<truncated, 500 chars>")

    def test_long_list_is_cut(self):
        out = summarize_payload(list("abcdefghijklmnopqrstuvwxyz"))
        assert len(out) == 21
        assert out[-1] == "<6 more items>"

    def test_arbitrary_object_replaced_by_type_name(self):
        class DoclingDocument:
            pass

        assert summarize_payload({"doc": DoclingDocument()}) == {"doc": "<DoclingDocument>"}

    def test_does_not_mutate_input(self):
        value = {"embeddings": [[0.1] * 10], "text": "y" * 300}
        snapshot = json.dumps(value)
        summarize_payload(value)
        assert json.dumps(value) == snapshot


class TestPayloadGuard:
    def test_shrinks_inputs_outputs_and_attributes(self):
        span = FakeSpan(
            inputs={"input": ["chunk text " * 100] * 5, "model": "m"},
            outputs={"data": [{"embedding": [0.1] * 1536, "index": i} for i in range(100)]},
            attributes={"vec": [0.3] * 1024, "file": "a.pdf", "mlflow.spanType": "UNKNOWN"},
        )
        payload_guard(span)

        assert span.inputs["model"] == "m"
        assert all(s.endswith("chars>") for s in span.inputs["input"])
        assert span.outputs["data"][0] == {"embedding": "<vector dim=1536>", "index": 0}
        assert span.attributes == {
            "vec": "<vector dim=1024>", "file": "a.pdf", "mlflow.spanType": "UNKNOWN"
        }
        total = len(json.dumps([span.inputs, span.outputs, span.attributes]))
        assert total < 3 * mlflow_tracing._MAX_PAYLOAD_CHARS

    def test_whole_payload_is_capped(self):
        span = FakeSpan(outputs={f"k{i}": "v" * 199 for i in range(500)})
        payload_guard(span)
        assert isinstance(span.outputs, str)
        assert len(span.outputs) < mlflow_tracing._MAX_PAYLOAD_CHARS + 100

    def test_no_payload_is_noop(self):
        span = FakeSpan(attributes={"chunk_count": 4})
        payload_guard(span)
        assert span.inputs is None and span.outputs is None
        assert span.attributes == {"chunk_count": 4}


class TestAnnotateActiveSpan:
    def test_noop_without_mlflow(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "mlflow", raising=False)
        mlflow_tracing.annotate_active_span(a=1)  # must not raise

    def test_sets_non_none_attributes(self, monkeypatch):
        span = FakeSpan()
        fake = SimpleNamespace(get_current_active_span=lambda: span)
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        mlflow_tracing.annotate_active_span(a=1, b=None)
        assert span.attributes == {"a": 1}


def test_embed_texts_span_carries_metrics_not_vectors(monkeypatch):
    def _create(**kwargs):
        data = [SimpleNamespace(index=i, embedding=[0.5] * 1536) for i in range(len(kwargs["input"]))]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=42, total_tokens=42))

    client = MagicMock()
    client.embeddings.create.side_effect = _create
    monkeypatch.setattr(embeddings_mod, "get_embedding_client", lambda *a: client)
    monkeypatch.setattr(embeddings_mod, "start_child_span", lambda *a, **k: "span")
    captured = {}
    monkeypatch.setattr(
        embeddings_mod, "end_span", lambda span, **attrs: captured.update(attrs)
    )

    vectors, errors = embeddings_mod.embed_texts(
        ["a", "b", "c"], "model", "http://x", "key", run_id="run", parent_span="parent"
    )

    assert len(vectors) == 3 and errors == []
    assert captured["text_count"] == 3
    assert captured["embedding_dimensions"] == 1536
    assert captured["api_calls"] == 1
    assert captured["retry_count"] == 0
    assert captured["prompt_tokens"] == 42
    assert captured["total_tokens"] == 42
    assert all(not isinstance(v, (list, tuple)) for v in captured.values())


def test_prompt_extraction_records_usage_only(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"chunk_size": 500}'))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(prompt_mod, "OpenAI", lambda **kw: client)
    annotations = {}
    monkeypatch.setattr(
        prompt_mod, "annotate_active_span", lambda **attrs: annotations.update(attrs)
    )

    result = prompt_mod.extract_config_from_prompt("use 500 chunks", "m", "http://x", "k")

    assert result["chunk_size"] == 500
    assert annotations == {
        "llm_prompt_tokens": 100,
        "llm_completion_tokens": 10,
        "llm_total_tokens": 110,
        "llm_response_parsed": True,
    }


def test_tracing_run_does_not_enable_autolog(monkeypatch):
    fake_mlflow = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    installed = []
    monkeypatch.setattr(
        "src.utils.install_payload_guard", lambda: installed.append(True)
    )
    monkeypatch.setattr(agent_mod, "_discover_paths", lambda settings: ([], 0))
    monkeypatch.setattr(agent_mod, "_invoke_pipeline", lambda *a, **k: [])

    agent_mod._run_workflow_with_mlflow({"user_prompt": "", "shard_index": 0, "shard_count": 1})

    assert installed == [True]
    fake_mlflow.openai.autolog.assert_not_called()
    fake_mlflow.langchain.autolog.assert_not_called()
    fake_mlflow.autolog.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
