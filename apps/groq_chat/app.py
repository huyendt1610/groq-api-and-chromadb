import streamlit as st
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")

    st.title("Groq Chat App")
    st.write("This is a simple chat app built with Streamlit and Groq's API. It uses Langchain to manage the conversation flow and memory.")

    st.sidebar.title("Customization")
    model = st.sidebar.selectbox("Select a model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"])

    conversation_memory_length = st.sidebar.slider("Conversation Memory Length", min_value=1, max_value=10, value=5)

    user_question = st.text_input("Ask a question to the Groq model:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'message_history' not in st.session_state:
        st.session_state.message_history = ChatMessageHistory()

    # Initialize the Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model=model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    def get_session_history(session_id: str) -> ChatMessageHistory:
        # Trim to last N turns (each turn = 1 human + 1 AI message)
        all_messages = st.session_state.message_history.messages
        trimmed = ChatMessageHistory()
        trimmed.messages = all_messages[-(conversation_memory_length * 2):]
        return trimmed

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    if user_question:
        response = chain_with_history.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": "streamlit_session"}}
        )
        st.session_state.message_history.add_user_message(user_question)
        st.session_state.message_history.add_ai_message(response.content)

        message = {"human": user_question, "AI": response.content}
        st.session_state.chat_history.append(message)
        st.write(f"**Groq:** {response.content}")

if __name__ == "__main__":
    main()
