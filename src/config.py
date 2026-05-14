"""
Configuration for PDF processing agent
"""

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 1536

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Document processing configuration
PDF_FOLDER_PATH = "./pdfs"