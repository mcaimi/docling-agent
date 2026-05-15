"""
Default Configuration for PDF processing agent
"""

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BASE_URL: str | None = None
EMBEDDING_API_KEY: str | None = None
OPENAI_API_KEY: str | None = None

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DOCLING_CHUNKING_STRATEGY = "fixed"

# Document processing configuration
PDF_FOLDER_PATH = "./pdfs"

# Docling PDF pipeline (standard or vlm)
DOCLING_PDF_PIPELINE_ENV = ""
DOCLING_PDF_PIPELINE_DEFAULT = "standard"

# Remote VLM (OpenAI-compatible chat completions URL)
DOCLING_VLM_API_URL_ENV = "DOCLING_VLM_API_URL"
DOCLING_VLM_API_HEADERS_ENV = "{}"
DOCLING_VLM_API_PARAMS_ENV = "{}"
DOCLING_VLM_PRESET_ENV = "smoldocling"
DOCLING_VLM_PRESET_DEFAULT = "smoldocling"

