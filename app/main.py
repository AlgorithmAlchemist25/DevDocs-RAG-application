import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from .schemas import (
    QueryRequest, TitleRequest, TitleResponse,
    ConversationRequest, ConversationUpdateRequest,
    DocumentResponse, DocumentListItem, DeleteDocumentResponse,
)
from .graph import chatbot
from .generation import generate_chat_title
from .conversations import (
    get_conversations,
    create_conversation,
    get_single_conversation,
    update_conversation,
    conversation_exists,
)
from .ingestion import ingest_document, delete_document, list_documents


app = FastAPI(
    title="DevDocs RAG API",
    description="RAG API for querying DevDocs knowledge base",
    version="0.1.0"
)


# Health check
@app.get("/health")
def check_health(): 
    return {'status':'healthy'}


# Send a query to the LLM for getting RAG based response
@app.post("/query")
async def send_query(request : QueryRequest):
    
    # Create the conversation in database and generate the name
    if not conversation_exists(request.thread_id):
        title = await generate_chat_title(request.query)
        
        create_conversation(
            request.thread_id,
            title
        )
    
    def generate_response():
        for message_chunk, metadata in chatbot.stream(
            {'messages' : [HumanMessage(content = request.query)]},
            config = {'configurable' : {
                'thread_id' : request.thread_id
                }
            },
            stream_mode = 'messages'
        ):
            if metadata.get('langgraph_node') == 'generate' and isinstance(message_chunk, AIMessage):
                if message_chunk.content:
                    yield message_chunk.content
            
    return StreamingResponse(
        generate_response(),
        media_type = 'text/plain'
    )


# Load the current conversations from langgraph state
@app.get("/conversation/{thread_id}")
def get_conversation(thread_id : str):  
    state = chatbot.get_state(
        config = {'configurable':{'thread_id':thread_id}}
    )
    
    messages = state.values.get('messages',[])
    
    return {'messages' : [
            {'type' : message.type,
             'content' : message.content}
            
            for message in messages
            ]
        }
    
    
# Generate a title for the current conversation based on the very first prompt given
@app.post("/conversation/title", response_model = TitleResponse)
async def generate_conversation_title(request : TitleRequest):
    query = request.query
    
    title = await generate_chat_title(query)
    
    return {'title':title}


# Create a new conversation in Supabase database
@app.post("/conversation")
def create_conversation_endpoint(request : ConversationRequest):
    
    result = create_conversation(
        request.thread_id,
        request.title
    )
    
    return {'conversation':result}


# Load all the conversations from supabase database for toggling through sidebar
@app.get("/conversations")
def get_conversations_endpoint():
    conversations = get_conversations()
    
    return {'conversations':conversations}


# Load a single conversation from supabase database
@app.get("/conversation/{thread_id}/metadata")
def get_conversation_endpoint(thread_id : str):
    
    conversation = get_single_conversation(thread_id)
    
    return {'conversation':conversation}


# Update a conversation name
@app.patch("/conversation/{thread_id}")
def update_conversation_endpoint(request : ConversationUpdateRequest):
    conversation = update_conversation(
        request.thread_id,
        request.title
    )
    
    return {'conversation':conversation}


# ----------------------------------------
# Document Upload & Management
# ----------------------------------------

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, chunk it, embed it, and store it in the vector store.
    Accepts only PDF files.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to a temp location so PyMuPDF can read it.
    # Use mkstemp so we control when the fd is closed (important on Windows).
    import os as _os
    fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
    tmp_path = Path(tmp_name)
    try:
        # Write contents and close the fd before PyMuPDF opens the file.
        contents = await file.read()
        with _os.fdopen(fd, "wb") as f:
            f.write(contents)
        fd = None  # fd is now closed

        result = ingest_document(tmp_path, file.filename)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # On Windows a process may still hold the file briefly; ignore lock errors.
        if fd is not None:
            try:
                _os.close(fd)
            except OSError:
                pass
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows lock — file will be cleaned up by the OS on next boot

    return DocumentResponse(
        document_id=result["document_id"],
        filename=file.filename,
        chunk_count=result["chunk_count"],
        message=f"Document '{file.filename}' ingested successfully with {result['chunk_count']} chunks.",
    )


@app.get("/documents")
def get_documents():
    """List all uploaded documents."""
    docs = list_documents()
    return {"documents": docs}


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def remove_document(document_id: int):
    """Delete a document and all its vector embeddings."""
    success = delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteDocumentResponse(
        success=True,
        message=f"Document {document_id} deleted successfully."
    )