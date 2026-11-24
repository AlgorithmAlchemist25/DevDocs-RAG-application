from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    thread_id: str


class TitleRequest(BaseModel):
    query: str


class TitleResponse(BaseModel):
    title: str


class ConversationRequest(BaseModel):
    thread_id: str
    title: str


class ConversationUpdateRequest(BaseModel):
    thread_id: str
    title: str


# --- Document upload / management schemas ---

class DocumentResponse(BaseModel):
    document_id: int
    filename: str
    chunk_count: int
    message: str


class DocumentListItem(BaseModel):
    id: int
    filename: str
    created_at: Optional[str] = None


class DeleteDocumentResponse(BaseModel):
    success: bool
    message: str