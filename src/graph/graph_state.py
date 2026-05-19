#!/usr/bin/env python

"""Shared LangGraph state schema for the PDF processing workflow."""

# import typing
from typing import Any, Literal, NotRequired, TypedDict


# define the state schema for the PDF processor
# this tracks the state of the agent workflow
class PDFProcessorState(TypedDict):
    # LLM settings for optional preprocessing
    user_prompt: NotRequired[str]
    llm_model: NotRequired[str]
    base_url: NotRequired[str]
    api_key: NotRequired[str]

    # discovered source documents folder path
    folder_path: str

    # embedding settings
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str

    # pipeline settings
    conversion_pipeline: NotRequired[Literal["standard", "vlm"]]
    chunking_strategy: NotRequired[Literal["fixed", "hybrid"]]

    # std pipeline settings
    embedding_dimensions: NotRequired[int]
    chunk_size: NotRequired[int]
    chunk_overlap: NotRequired[int]

    # chunked documents
    documents: list[dict[str, Any]]

    # VLM pipeline settings
    vlm_api_url: NotRequired[str]
    vlm_api_headers: NotRequired[dict[str, str]]
    vlm_api_params: NotRequired[dict[str, Any]]
    vlm_preset: NotRequired[str]

    # input
    pdf_files: list[str]

    # output
    embeddings: list[list[float] | None]


# export the state schema
__all__ = ["PDFProcessorState"]
