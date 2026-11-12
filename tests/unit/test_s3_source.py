#!/usr/bin/env python
"""Unit tests for S3 document discovery and fetching (mocked with moto)."""

import boto3
import pytest
from moto import mock_aws

import src.graph.document_pipeline as document_pipeline_mod
from src.docling_agent import EXIT_CONFIG_ERROR, main, process_documents
from src.graph import s3_source
from src.graph.document_paths import (
    DOCUMENT_GLOBS,
    discover_document_paths,
    shard_paths,
    validate_source_settings,
)
from src.graph.s3_source import download_s3_object, parse_s3_uri

BUCKET = "docs"


@pytest.fixture
def s3(monkeypatch):
    """A mocked S3 with an empty bucket; clients are rebuilt per test."""
    for key, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_DEFAULT_REGION", "us-east-1"),
    ):
        monkeypatch.setenv(key, value)
    s3_source._clients.clear()
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client
    s3_source._clients.clear()


def _settings(**overrides):
    return {
        "source_backend": "s3",
        "s3_bucket": BUCKET,
        "s3_prefix": "",
        "s3_recursive": 0,
        "s3_region": "us-east-1",
        **overrides,
    }


def _put(client, *keys, body=b"x"):
    for key in keys:
        client.put_object(Bucket=BUCKET, Key=key, Body=body)


def _markdown(i):
    return (
        f"# Document {i}\n\nSome content for document {i}, long enough "
        f"to produce more than one chunk with a small chunk_size.\n\n"
        f"## Section\n\nMore padding text for document {i}."
    ).encode()


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("s3://bucket/key.pdf", ("bucket", "key.pdf")),
        ("s3://bucket/a/b/c.md", ("bucket", "a/b/c.md")),
        ("s3://bucket", ("bucket", "")),
    ],
)
def test_parse_s3_uri(uri, expected):
    assert parse_s3_uri(uri) == expected


@pytest.mark.parametrize("uri", ["/local/file.pdf", "s3:///key.pdf", "https://x/y.pdf"])
def test_parse_s3_uri_rejects_invalid(uri):
    with pytest.raises(ValueError):
        parse_s3_uri(uri)


def test_listing_is_sorted_filtered_and_non_recursive(s3):
    _put(s3, "in/b.pdf", "in/a.md", "in/c.docx", "in/ignored.txt", "in/UPPER.PDF",
         "in/nested/d.pdf", "in/", "other/e.pdf")
    paths = discover_document_paths(_settings(s3_prefix="in"))
    assert paths == ["s3://docs/in/a.md", "s3://docs/in/b.pdf", "s3://docs/in/c.docx"]


def test_recursive_listing_includes_nested_prefixes(s3):
    _put(s3, "in/b.pdf", "in/nested/d.pdf", "in/nested/deeper/e.md", "in/nested/", "other/f.pdf")
    paths = discover_document_paths(_settings(s3_prefix="in/", s3_recursive=1))
    assert paths == [
        "s3://docs/in/b.pdf",
        "s3://docs/in/nested/d.pdf",
        "s3://docs/in/nested/deeper/e.md",
    ]


def test_empty_prefix_lists_bucket_root(s3):
    _put(s3, "root.md", "sub/nested.md")
    assert discover_document_paths(_settings()) == ["s3://docs/root.md"]


def test_listing_paginates(s3):
    keys = [f"many/{i:04d}.md" for i in range(1005)]
    _put(s3, *keys)
    paths = discover_document_paths(_settings(s3_prefix="many/"))
    assert paths == [f"s3://docs/{k}" for k in keys]


def test_s3_paths_shard_without_overlap(s3):
    _put(s3, *(f"{i:02d}.pdf" for i in range(10)))
    paths = discover_document_paths(_settings())
    shards = [shard_paths(paths, i, 3) for i in range(3)]
    flat = [p for shard in shards for p in shard]
    assert sorted(flat) == paths
    assert len(flat) == len(set(flat))


def test_missing_bucket_fails_listing(s3):
    with pytest.raises(Exception):
        discover_document_paths(_settings(s3_bucket="does-not-exist"))


def test_local_backend_is_unchanged(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.md").write_text("x")
    # s3_recursive has no effect on local folders
    paths = discover_document_paths(
        {"source_backend": "local", "folder_path": str(tmp_path), "s3_recursive": 1}
    )
    assert paths == [str(tmp_path / "a.md")]


@pytest.mark.parametrize(
    "settings",
    [
        {"source_backend": "s3", "s3_bucket": None},
        {"source_backend": "s3", "s3_bucket": ""},
        {"source_backend": "ftp"},
    ],
)
def test_invalid_source_settings_rejected(settings):
    with pytest.raises(ValueError):
        validate_source_settings(settings)


def test_cli_exits_with_config_error_without_bucket(monkeypatch):
    monkeypatch.setenv("SOURCE_BACKEND", "s3")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setattr("sys.argv", ["docling_agent", "--envfile", "/dev/null"])
    assert main() == EXIT_CONFIG_ERROR


def test_download_keeps_basename(s3, tmp_path):
    _put(s3, "in/report.final.pdf", body=b"%PDF-data")
    target = download_s3_object("s3://docs/in/report.final.pdf", tmp_path, _settings())
    assert target == tmp_path / "report.final.pdf"
    assert target.read_bytes() == b"%PDF-data"


def test_client_is_cached_per_connection_settings(s3):
    a = s3_source.get_s3_client(_settings())
    assert s3_source.get_s3_client(_settings()) is a
    assert s3_source.get_s3_client(_settings(s3_addressing_style="path")) is not a


def _fake_embed_texts(texts, model, base_url, api_key, dimensions, run_id=None, parent_span=None):
    return [[0.1, 0.2, 0.3] for _ in texts], []


def test_process_documents_from_s3(s3, monkeypatch):
    monkeypatch.setattr(document_pipeline_mod, "embed_texts", _fake_embed_texts)
    for i in range(2):
        _put(s3, f"in/doc{i}.md", body=_markdown(i))
    # listed, but the object is gone by the time it is fetched
    _put(s3, "in/vanished.md")
    real_download = s3_source.download_s3_object

    def flaky_download(uri, dest_dir, settings):
        if uri.endswith("vanished.md"):
            s3.delete_object(Bucket=BUCKET, Key="in/vanished.md")
        return real_download(uri, dest_dir, settings)

    monkeypatch.setattr(document_pipeline_mod, "download_s3_object", flaky_download)

    result = process_documents(
        _settings(
            s3_prefix="in/",
            folder_path="/nonexistent",
            user_prompt="",
            conversion_pipeline="standard",
            chunking_strategy="fixed",
            chunk_size=100,
            chunk_overlap=10,
            embedding_model="text-embedding-ada-002",
            embedding_base_url="https://api.openai.com/v1",
            embedding_api_key="sk-fake",
            embedding_dimensions=None,
            vector_store_backend="none",
            max_concurrency=2,
            shard_index=0,
            shard_count=1,
        )
    )

    file_paths = {d["metadata"]["file_path"] for d in result["documents"]}
    assert file_paths == {"s3://docs/in/doc0.md", "s3://docs/in/doc1.md"}
    titles = {d["metadata"]["title"] for d in result["documents"]}
    assert titles == {"doc0", "doc1"}
    assert len(result["embeddings"]) == len(result["documents"])
    assert [(e["path"], e["stage"]) for e in result["error_reports"]] == [
        ("s3://docs/in/vanished.md", "pipeline")
    ]


def test_document_globs_are_the_s3_filter():
    # the contract's globs drive S3 filtering too, so launchers count the same way
    assert s3_source._matches("a/b/report.pdf", DOCUMENT_GLOBS)
    assert not s3_source._matches("a/b/notes.txt", DOCUMENT_GLOBS)
