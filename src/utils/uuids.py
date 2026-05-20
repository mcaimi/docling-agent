#!/usr/bin/env python
"""UUID utilities."""

import uuid
import time

def generate_uuid() -> str:
    """Generate a new UUID."""
    return str(uuid.uuid4())

def is_valid_uuid(uuid: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(uuid)
        return True
    except ValueError:
        return False

# generate a random mlflow run name from current timestamp and a random uuid
def generate_mlflow_run_name() -> str:
    """Generate a random MLflow run name from current timestamp and a random UUID."""
    return f"docling_document_processing_{time.time()}_{generate_uuid()}"

# export the functions
__all__ = [
    "generate_uuid",
    "is_valid_uuid",
    "generate_mlflow_run_name",
]