import os
import tempfile
from pathlib import Path

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import embedding_model
from .vector_store import supabase


# --------------------------------------------------
# Configuration
# --------------------------------------------------

headers_to_split_on = [
    ("#", "Main_Topic"),
    ("##", "Sub_Section"),
    ("###", "Nested_Topic"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

size_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=75
)


# --------------------------------------------------
# Ingest a single PDF (given its path on disk)
# --------------------------------------------------

def ingest_document(file_path: Path, original_filename: str) -> dict:
    """
    Load, chunk, embed and store one PDF document.

    Returns a dict with document_id and chunk_count.
    Raises an exception if anything goes wrong (caller should handle it).
    """

    # 1. Load document → markdown
    loader = PyMuPDF4LLMLoader(
        file_path=str(file_path),
        mode="single"
    )
    document = loader.load()
    markdown_document = document[0].page_content

    # 2. Structural splitting (by headers)
    structural_chunks = markdown_splitter.split_text(markdown_document)

    # 3. Size-based splitting
    final_chunks = size_text_splitter.split_documents(structural_chunks)

    # 4. Create document record in Supabase
    document_response = (
        supabase
        .table("documents")
        .insert({"filename": original_filename})
        .execute()
    )
    document_id = document_response.data[0]["id"]

    # 5. Generate embeddings for all chunks
    texts = [chunk.page_content for chunk in final_chunks]
    embeddings = embedding_model.embed_documents(texts)

    # 6. Build rows and bulk-insert into document_chunks
    rows = [
        {
            "document_id": document_id,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding,
        }
        for chunk, embedding in zip(final_chunks, embeddings)
    ]

    supabase.table("document_chunks").insert(rows).execute()

    return {"document_id": document_id, "chunk_count": len(rows)}


# --------------------------------------------------
# Delete a document and all its chunks
# --------------------------------------------------

def delete_document(document_id: int) -> bool:
    """
    Remove a document record and its chunks from Supabase.
    Chunk deletion relies on the ON DELETE CASCADE constraint;
    if that is not set up, chunks are deleted explicitly first.
    """
    # Delete chunks first (safe even with CASCADE)
    supabase.table("document_chunks").delete().eq("document_id", document_id).execute()

    # Delete the document record
    response = (
        supabase.table("documents")
        .delete()
        .eq("id", document_id)
        .execute()
    )

    return len(response.data) > 0


# --------------------------------------------------
# List all ingested documents
# --------------------------------------------------

def list_documents() -> list:
    response = (
        supabase.table("documents")
        .select("id, filename, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data
