#!/usr/bin/env python
"""S3 helpers for discovering and fetching processable documents.

Documents are addressed as ``s3://bucket/key`` strings throughout the
pipeline, so sorting, sharding, vector IDs and error reports work the same as
for local paths; only discovery and conversion need to know about S3.
"""

import fnmatch
import logging
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

logger = logging.getLogger(__name__)

S3_SCHEME = "s3://"

# Settings that shape the client; one cached client per distinct combination.
_CLIENT_KEYS = (
    "s3_endpoint_url",
    "s3_region",
    "s3_access_key_id",
    "s3_secret_access_key",
    "s3_session_token",
    "s3_addressing_style",
    "s3_verify_ssl",
    "s3_ca_bundle",
)

# boto3 clients are thread-safe, so concurrent conversion tasks share one.
_clients: dict[tuple, Any] = {}
_clients_lock = threading.Lock()


def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith(S3_SCHEME)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    if not is_s3_uri(uri):
        raise ValueError(f"not an S3 URI: {uri!r}")
    bucket, _, key = uri[len(S3_SCHEME):].partition("/")
    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {uri!r}")
    return bucket, key


def require_boto3():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        # a missing optional dependency is a configuration problem (exit 2)
        raise ValueError(
            "SOURCE_BACKEND=s3 requires boto3: install the 's3' extra"
        ) from e
    return boto3, Config


def get_s3_client(settings: dict[str, Any]):
    """Return a cached S3 client for the connection settings in ``settings``."""
    cache_key = tuple(settings.get(k) for k in _CLIENT_KEYS)
    with _clients_lock:
        client = _clients.get(cache_key)
        if client is not None:
            return client

        boto3, Config = require_boto3()
        verify: bool | str = bool(settings.get("s3_verify_ssl", 1))
        if verify and settings.get("s3_ca_bundle"):
            verify = settings["s3_ca_bundle"]
        # Unset credentials fall through to boto3's default chain.
        client = boto3.session.Session().client(
            "s3",
            endpoint_url=settings.get("s3_endpoint_url") or None,
            region_name=settings.get("s3_region") or None,
            aws_access_key_id=settings.get("s3_access_key_id") or None,
            aws_secret_access_key=settings.get("s3_secret_access_key") or None,
            aws_session_token=settings.get("s3_session_token") or None,
            verify=verify,
            config=Config(
                retries={"mode": "standard"},
                s3={"addressing_style": settings.get("s3_addressing_style") or "auto"},
            ),
        )
        _clients[cache_key] = client
        return client


def _matches(key: str, patterns: Iterable[str]) -> bool:
    name = PurePosixPath(key).name
    # fnmatchcase: same case-sensitive matching as Path.glob on Linux
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def list_s3_document_paths(
    bucket: str,
    prefix: str,
    settings: dict[str, Any],
    patterns: Iterable[str],
    recursive: bool = False,
) -> list[str]:
    """Return sorted ``s3://bucket/key`` URIs of objects matching ``patterns``.

    ``prefix`` is treated as a folder: a trailing ``/`` is added when missing.
    Without ``recursive`` only keys directly under it are listed, matching the
    non-recursive local glob.
    """
    if not bucket:
        raise ValueError("SOURCE_BACKEND=s3 requires S3_BUCKET")
    prefix = prefix or ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    patterns = tuple(patterns)

    params: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        params["Delimiter"] = "/"

    client = get_s3_client(settings)
    paths: list[str] = []
    for page in client.get_paginator("list_objects_v2").paginate(**params):
        for obj in page.get("Contents", ()):
            key = obj["Key"]
            # skip "folder" placeholder objects
            if key.endswith("/") or not _matches(key, patterns):
                continue
            paths.append(f"{S3_SCHEME}{bucket}/{key}")
    paths.sort()
    return paths


def download_s3_object(uri: str, dest_dir: str | Path, settings: dict[str, Any]) -> Path:
    """Download ``uri`` into ``dest_dir``, keeping the object's basename.

    Docling detects the input format from the file extension, and the record
    title comes from the file stem, so the name must be preserved.
    """
    bucket, key = parse_s3_uri(uri)
    name = PurePosixPath(key).name
    if not name:
        raise ValueError(f"S3 URI does not name an object: {uri!r}")
    target = Path(dest_dir) / name
    get_s3_client(settings).download_file(bucket, key, str(target))
    return target


__all__ = [
    "S3_SCHEME",
    "is_s3_uri",
    "parse_s3_uri",
    "require_boto3",
    "get_s3_client",
    "list_s3_document_paths",
    "download_s3_object",
]
