
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


# ----------------------------------------
# 2. Session Setup
# ----------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'conversations' not in st.session_state:
    st.session_state['conversations'] = load_conversations()


# ----------------------------------------
# 3. Sidebar UI
# ----------------------------------------

st.sidebar.title('DevDocs')
st.sidebar.header('My Conversations')

if st.sidebar.button('New Chat'):
    reset_chat()

for conversation in st.session_state['conversations']:

    thread_id = conversation['thread_id']
    title = conversation['title']

    if st.sidebar.button(title):

        st.session_state['thread_id'] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:

            if message['type'] == 'human':
                role = 'user'
            else:
                role = 'assistant'

            temp_messages.append({
                'role': role,
                'content': message['content']
            })

        st.session_state['message_history'] = temp_messages

        st.rerun()


# ----------------------------------------
# 4. Main UI
# ----------------------------------------

for message in st.session_state['message_history']:

    with st.chat_message(message['role']):
        st.text(message['content'])


# Input Box

user_input = st.chat_input(
    'Ask a question about your documentation'
)

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
        st.text(user_input)

    try:

        with st.chat_message('assistant'):

            response = requests.post(
                f"{API_URL}/query",
                json={
                    'query': user_input,
                    'thread_id': str(
                        CONFIG['configurable']['thread_id']
                    )
                },
                stream=True,
                timeout=(10, 180)
            )

            if response.status_code == 200:

                ai_message = st.write_stream(
                    response.iter_content(
                        chunk_size=None,
                        decode_unicode=True
                    )
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
