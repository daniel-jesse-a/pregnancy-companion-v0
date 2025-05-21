# At the top of your app.py file
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Continue with standard sqlite3
    pass

import streamlit as st
import os
import uuid

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

import rag_methods


MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

st.set_page_config(
    page_title="Pregnancy Companion V0",
    page_icon="🤰",
    layout="centered",
    initial_sidebar_state="expanded"
)

#-----Header-----
st.html("""<h2 style=text-align: center;"> 🤰 Pregnancy Companion V0.0 </h2>""")

# ---- Initial Setup ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]


# Add this initialization for the model
if "model" not in st.session_state:
    st.session_state.model = MODELS[0]  # Default to the first model

# Add a model selector in the sidebar
with st.sidebar:
    st.divider()
    st.selectbox("Select Model", MODELS, key="model")
    
    cols0 = st.columns(2)
    with cols0[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")


if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

# Add this to initialize the vector store
if not st.session_state.vector_store_initialized:
    st.session_state.vector_store_initialized = rag_methods.initialize_vector_store()

#Main Chat App

model_provider = st.session_state.model.split("/")[0]
if model_provider == "openai":
    # Get API key from rag_methods
    api_key = rag_methods.get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not found. Please add it to your Streamlit secrets.")
    else:
        llm_stream = ChatOpenAI(
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
            openai_api_key=api_key
        )

for message in st.session_state.messages:  # Changed 'messages' to 'message' to avoid variable shadowing
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create a placeholder for the response
        message_placeholder = st.empty()
        full_response = ""
        
        # Get the model name from session state
        model_name = st.session_state.model.split("/")[-1]
        
        # Stream the response and build the full response
        for chunk in rag_methods.get_rag_response(prompt, st.session_state.messages, model_name):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        # Replace the placeholder with the final response
        message_placeholder.markdown(full_response)
        
        # Save the assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
