#!/usr/bin/env python

"""Shared LangGraph state schema for the PDF processing workflow."""

# import typing
from typing import Any, Literal, NotRequired, TypedDict

# define the state schema for the PDF processor
# this tracks the state of the agent workflow
class PDFProcessorState(TypedDict):
    folder_path: str
    api_key: str
    embedding_model: str
    connection_params: dict
    chunk_size: NotRequired[int]
    chunk_overlap: NotRequired[int]
    pdf_files: list[str]
    documents: list[dict[str, Any]]
    embeddings: list[list[float] | None]
    embedding_model: str
    conversion_pipeline: NotRequired[Literal["standard", "vlm"]]
    vlm_api_url: NotRequired[str]
    vlm_api_headers: NotRequired[dict[str, str]]
    vlm_api_params: NotRequired[dict[str, Any]]
    vlm_preset: NotRequired[str]

# export the state schema
__all__ = ["PDFProcessorState"]