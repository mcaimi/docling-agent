#!/usr/bin/env python
"""Unit tests for document discovery and sharding."""

import pytest

from src.graph.document_paths import list_document_paths, shard_paths


def test_list_document_paths_is_sorted_and_filtered(tmp_path):
    for name in ("b.pdf", "a.md", "c.docx", "ignored.txt"):
        (tmp_path / name).write_text("x")
    paths = list_document_paths(tmp_path)
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["a.md", "b.pdf", "c.docx"]


@pytest.mark.parametrize("count", [1, 2, 3, 7, 12])
def test_shards_cover_everything_without_overlap(count):
    paths = [f"/docs/{i:02d}.pdf" for i in range(10)]
    shards = [shard_paths(paths, i, count) for i in range(count)]
    flat = [p for shard in shards for p in shard]
    assert sorted(flat) == paths
    assert len(flat) == len(set(flat))


def test_shards_are_balanced():
    paths = [str(i) for i in range(10)]
    sizes = [len(shard_paths(paths, i, 3)) for i in range(3)]
    assert sizes == [4, 3, 3]


@pytest.mark.parametrize("index, count", [(0, 0), (-1, 2), (2, 2), (5, 3)])
def test_invalid_shard_assignment_rejected(index, count):
    with pytest.raises(ValueError):
        shard_paths(["a"], index, count)
