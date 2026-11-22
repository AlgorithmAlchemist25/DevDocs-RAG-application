
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# ----------------------------
# 1. Response Generation
# ----------------------------

model1 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=2048,
    reasoning_effort= "low"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict, factual GenAI Expert and AI Engineer.

        Answer the user's question using these rules:

        1. Use the retrieved context first.
        2. If the retrieved context does not contain the answer, use the conversation history.
        3. If neither contains the complete answer, reply exactly:
        "I don't know."

        Do not use outside knowledge.
        Do not assume or fill gaps.
        Keep your response concise, factual, and direct."""
    ),

    MessagesPlaceholder(variable_name="history"),

    (
        "human",
        """[RETRIEVED CONTEXT]
        {context}

        [USER QUERY]
        {user_query}"""
    )
])

chain = prompt | model1


def generate(query: str, history: list[BaseMessage], context: str):
    """Streams the response as LangChain AIMessageChunk objects."""

    return chain.stream({
        "user_query": query,
        "history": history,
        "context": context
    })


# ----------------------------
# 2. Chat Title Generation
# ----------------------------

model2 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=128,
    reasoning_effort="low"
)


async def generate_chat_title(query: str):

    messages = [
        SystemMessage(
            content=(
                "Given the conversation history, provide a SHORT name "
                "for the conversation. Focus on important keywords. "
                "Use the same language as the user's language. "
                "Output only the name. Never use more than 5 words."
            )
        ),
        HumanMessage(content=query)
    ]

    title = (await model2.ainvoke(messages)).content

    return title
