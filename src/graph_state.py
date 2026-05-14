"""Shared LangGraph state schema for the PDF processing workflow."""

from typing import Any, Literal, NotRequired, TypedDict


class PDFProcessorState(TypedDict):
    folder_path: str
    pdf_files: list[str]
    documents: list[dict[str, Any]]
    embeddings: list[list[float] | None]
    conversion_pipeline: NotRequired[Literal["standard", "vlm"]]
