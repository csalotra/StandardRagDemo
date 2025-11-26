import os
from pathlib import Path
from typing import List, Any

# LangChain loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredXMLLoader,
)

# Supported extensions
SUPPORTED_EXTS = {
    ".txt",
    ".pdf",
    ".md",
    ".docx",
    ".csv",
    ".json",
    ".xml",
    ".sql"
}

# Mapping extensions → loaders
LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".xml": UnstructuredXMLLoader,
    ".sql": TextLoader
}
    

def load_data(directory_path: str) -> List[Any]:
    """
    Load ALL supported documents from a directory recursively.

    Args:
        directory_path: Path to the directory containing documents.

    Returns:
        List of LangChain Document objects.
    """
    directory = Path(directory_path).resolve()
    print(f"[DEBUG] Resolved data path: {directory}")

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    all_docs = []

    # Recursively walk through all files
    for file_path in directory.rglob("*"):
        # print(f"[DEBUG] Found path: {file_path}")

        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        # select supported extensions
        if ext not in SUPPORTED_EXTS:
            print(f"[SKIPPED] Unsupported file: {file_path.name}")
            continue
        
        loader_class = LOADER_MAP.get(ext)

        try:
            # Select appropriate loader
            loader = loader_class(str(file_path))
            loaded_docs = loader.load()
            all_docs.extend(loaded_docs)
            print(f"Loaded {len(loaded_docs)} docs from {file_path.name}")

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_docs)}\n")
    return all_docs
