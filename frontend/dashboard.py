
import requests
import streamlit as st
from uuid import uuid4
import os

API_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
).rstrip("/")


# ----------------------------------------
# 1. Utility Functions
# ----------------------------------------

def generate_thread_id():
    return str(uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['conversations']:
        st.session_state['conversations'].append(thread_id)


def load_conversation(thread_id):
    try:
        response = requests.get(
            f"{API_URL}/conversation/{thread_id}",
            timeout=30
        )
        if response.status_code != 200:
            return []
        return response.json()['messages']
    except requests.RequestException as e:
        st.error(f"Could not load conversation: {e}")
        return []


def load_conversations():
    try:
        response = requests.get(
            f"{API_URL}/conversations",
            timeout=30
        )
        if response.status_code != 200:
            return []
        return response.json()['conversations']
    except requests.RequestException as e:
        st.error(f"Could not load conversations: {e}")
        return []


def load_documents():
    try:
        response = requests.get(f"{API_URL}/documents", timeout=30)
        if response.status_code != 200:
            return []
        return response.json().get('documents', [])
    except requests.RequestException as e:
        st.error(f"Could not load documents: {e}")
        return []


def upload_document(file):
    try:
        response = requests.post(
            f"{API_URL}/documents/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=(10, 300),   # 5 min for large PDFs
        )
        return response
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None


def delete_document(document_id):
    try:
        response = requests.delete(
            f"{API_URL}/documents/{document_id}",
            timeout=30
        )
        return response
    except requests.RequestException as e:
        st.error(f"Could not delete document: {e}")
        return None


# ----------------------------------------
# 2. Session Setup
# ----------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'conversations' not in st.session_state:
    st.session_state['conversations'] = load_conversations()

if 'documents' not in st.session_state:
    st.session_state['documents'] = load_documents()


# ----------------------------------------
# 3. Sidebar UI
# ----------------------------------------

st.sidebar.title('DevDocs')

# ---- Document Upload Section ----
st.sidebar.header('📄 Upload Document')

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF to chat with",
    type=["pdf"],
    help="Upload a PDF document. It will be processed and added to the knowledge base.",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    if st.sidebar.button("⬆️ Ingest Document", use_container_width=True):
        with st.sidebar.status(f"Processing **{uploaded_file.name}**…", expanded=True) as status:
            st.write("Uploading and chunking…")
            resp = upload_document(uploaded_file)

            if resp is not None and resp.status_code == 200:
                data = resp.json()
                st.write(f"✅ Done! Created **{data['chunk_count']}** chunks.")
                status.update(label="Ingestion complete!", state="complete")
                # Refresh document list
                st.session_state['documents'] = load_documents()
                st.rerun()
            else:
                detail = resp.json().get('detail', resp.text) if resp else "Unknown error"
                st.write(f"❌ Failed: {detail}")
                status.update(label="Ingestion failed", state="error")

# ---- Ingested Documents Section ----
st.sidebar.divider()
st.sidebar.header('🗂️ Ingested Documents')

if st.sidebar.button("🔄 Refresh", key="refresh_docs"):
    st.session_state['documents'] = load_documents()
    st.rerun()

if st.session_state['documents']:
    for doc in st.session_state['documents']:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.caption(f"📄 {doc['filename']}")
        with col2:
            if st.button("🗑️", key=f"del_{doc['id']}", help="Delete this document"):
                resp = delete_document(doc['id'])
                if resp and resp.status_code == 200:
                    st.sidebar.success(f"Deleted '{doc['filename']}'")
                    st.session_state['documents'] = load_documents()
                    st.rerun()
                else:
                    st.sidebar.error("Delete failed.")
else:
    st.sidebar.info("No documents uploaded yet.")

# ---- Conversations Section ----
st.sidebar.divider()
st.sidebar.header('💬 My Conversations')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    reset_chat()

for conversation in st.session_state['conversations']:
    thread_id = conversation['thread_id']
    title = conversation['title']

    if st.sidebar.button(title, key=f"conv_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            role = 'user' if message['type'] == 'human' else 'assistant'
            temp_messages.append({'role': role, 'content': message['content']})
        st.session_state['message_history'] = temp_messages
        st.rerun()


# ----------------------------------------
# 4. Main Chat UI
# ----------------------------------------

st.title("💬 DevDocs Chat")
st.caption("Ask questions about your uploaded documents.")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Input Box
user_input = st.chat_input('Ask a question about your documentation')

if user_input:
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    CONFIG = {
        'configurable': {
            'thread_id': st.session_state['thread_id']
        }
    }

    with st.chat_message('user'):
        st.markdown(user_input)

    try:
        with st.chat_message('assistant'):
            response = requests.post(
                f"{API_URL}/query",
                json={
                    'query': user_input,
                    'thread_id': str(CONFIG['configurable']['thread_id'])
                },
                stream=True,
                timeout=(10, 180)
            )

            if response.status_code == 200:
                ai_message = st.write_stream(
                    response.iter_content(chunk_size=None, decode_unicode=True)
                )

                st.session_state['message_history'].append({
                    'role': 'assistant',
                    'content': ai_message
                })

                st.session_state['conversations'] = load_conversations()
                st.rerun()

            else:
                st.error(
                    f"Backend error: {response.status_code} - "
                    f"{response.text}"
                )

    except requests.RequestException as e:
        st.error(f"Could not connect to backend: {e}")
