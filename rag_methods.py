#rag_methods.py

import streamlit as st
import os
import dotenv
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
# Use the community version for now since you're having issues with langchain-chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv(verbose=True)

# Check if API key is available
def get_openai_api_key():
    """Get OpenAI API key from secrets or environment variables"""
    
    api_key = None
    
    # First try to get from Streamlit secrets
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        print("Using API key from Streamlit secrets")
    
    # Then try environment variables
    elif "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
        print("Using API key from environment variables")
    
    # Check if the API key is valid (basic format check)
    if api_key and (not api_key.startswith("sk-") or len(api_key) < 30):
        st.error(f"Invalid API key format. OpenAI API keys should start with 'sk-'")
        return None
        
    # If no API key found
    if not api_key:
        st.error("OpenAI API key not found. Please add it to your Streamlit secrets or environment variables.")
        return None
        
    return api_key
    
openai_api_key = get_openai_api_key()

DB_DOCS_LIMIT = 100
PERSIST_DIRECTORY = "chroma_db"
DOCS_DIRECTORY = "docs"

# --- Streaming Response Function ---
def stream_llm_response(llm_stream, messages):
    response_message = ""
    
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content

    st.session_state.messages.append({"role": "assistant", "content": response_message})

# --- Document Loading Functions ---
def load_documents_from_directory(directory_path=DOCS_DIRECTORY):
    """Load documents from the docs directory"""
    documents = []
    directory = Path(directory_path)
    
    if not directory.exists():
        st.error(f"Directory not found: {directory_path}")
        return []
    
    for file_path in directory.glob("**/*"):
        if file_path.is_file():
            try:
                file_extension = file_path.suffix.lower()
                
                if file_extension == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif file_extension == ".docx":
                    loader = Docx2txtLoader(str(file_path))
                elif file_extension in [".txt", ".md"]:
                    loader = TextLoader(str(file_path))
                else:
                    continue  # Skip unsupported file types
                    
                documents.extend(loader.load())
                st.success(f"Loaded: {file_path.name}")
                
            except Exception as e:
                st.error(f"Error loading document {file_path.name}: {str(e)}")
    
    return documents

# --- Document Processing Functions ---
def split_documents(documents, chunk_size=10000, chunk_overlap=8000):
    """Split documents into chunks for processing"""
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(documents)

# --- Vector Store Functions ---
def get_vector_store(documents=None):
    """Create or load a vector store from documents"""
    api_key = get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key is required for embeddings.")
        return None
        
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    if documents:
        # Create a new vector store from documents
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        return vector_store
    elif os.path.exists(PERSIST_DIRECTORY):
        # Load existing vector store
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        # No documents provided and no existing DB
        return None

# --- Simplified RAG Implementation ---
def get_rag_chain(llm):
    """Create a simple RAG chain without history awareness"""
    vector_store = get_vector_store()
    
    if not vector_store:
        return None
    
    # Create a retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    # Create a template for the prompt
    template = """You are a helpful pregnancy companion assistant. Answer the question based on the following context.
    If you don't know the answer or can't find it in the context, say so and provide general information if possible.
    When using information not found in the context, include a note like "[Note: This information comes from my general knowledge]".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    # Create a simple RAG chain
    from langchain_core.runnables import RunnablePassthrough
    
    # Format the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Streamlit Interface Functions ---
def initialize_vector_store():
    """Initialize the vector store with documents from the docs directory"""
    # Check if vector store already exists
    if os.path.exists(PERSIST_DIRECTORY):
        st.sidebar.success("Using existing document database")
        return True
    
    # If not, create it from the docs directory
    with st.sidebar:
        with st.spinner("Processing documents from docs directory..."):
            docs = load_documents_from_directory()
            
            if docs:
                # Split documents and create vector store
                chunks = split_documents(docs)
                if chunks:
                    vector_store = get_vector_store(chunks)
                    st.success(f"Successfully processed {len(docs)} documents into {len(chunks)} chunks!")
                    return True
            else:
                st.error("No documents were found in the docs directory.")
    
    return False

def get_rag_response(user_input, chat_history, model_name):
    """Get a response from the RAG system using Ditta Depner's philosophy"""
    try:
        # Get API key
        api_key = get_openai_api_key()
        if not api_key:
            yield "OpenAI API key not found. Please add it to your Streamlit secrets or environment variables."
            return
            
        # Initialize the LLM
        if model_name.startswith("gpt"):
            llm = ChatOpenAI(
                model_name=model_name, 
                temperature=0.3, 
                streaming=True,
                openai_api_key=api_key
            )
            
        # Get the vector store
        vector_store = get_vector_store()
        
        if not vector_store:
            st.warning("No document database found. Please check the docs directory.")
            yield "I'm sorry, I couldn't find any documents to search through. Please make sure there are documents in the docs directory."
            return
        
        # Create a retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Increased to 5 for better context
        
        # Get relevant documents first (non-streaming)
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Enhanced prompt with emotional intelligence and clear boundaries
        prompt = f"""You are the Pregnancy Companion AI, a gentle and emotionally aware assistant trained exclusively on the teachings of Ditta Depner. Your tone reflects empathy, calmness, and deep presence. Your goal is to offer support, reflection, and guidance based **only** on the content provided below.

        Your identity is not general-purpose AI. You are trained in Ditta's holistic approach to pregnancy, fertility, birth, and postpartum care. Your responses should reflect:
        - Trauma-informed awareness
        - A soft, human-like tone
        - Ditta's voice, style, and emotional sensitivity
        - Clear boundaries around non-medical guidance

        Context (your only source of information):
        {context}

        User's Question:
        {user_input}

        Previous Conversation:
        {format_chat_history(chat_history)}

        Instructions:
        1. You must only answer using the provided context above.
        2. If the information isn't found in the context, say:
        "I'm not able to offer that specific information right now, but I'm here to support you however I can."
        (You may add a note from general knowledge **only** if absolutely necessary, and mark it clearly like: "[Note: General insight]")
        3. NEVER offer medical advice. If the user asks something medical, say:
        "I'm here to support your emotional and personal journey, but I can't provide medical guidance. It's best to speak to a healthcare professional."
        4. Always respond in a gentle, nurturing, emotionally grounded tone—as if you were a trusted companion during a vulnerable time.
        5. Never invent, assume, or generalize beyond what's clearly written in the context.
        6. Where possible, reference where the answer is drawn from (e.g., "In Ditta's fertility book..." or "In the course section on birth fears...").
        7. If the user asks how you work, say:
        "I've been trained on Ditta's teachings and can only share what's already known in that body of wisdom."
        8. Consider the conversation history for context, but prioritize the current question.

        Answer (gently, from the context above):
        """
        
        # Stream the response directly from the LLM
        for chunk in llm.stream([{"role": "user", "content": prompt}]):
            yield chunk.content
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        yield f"I'm sorry, I encountered an error while processing your request. Please try again or ask a different question. Error details: {str(e)}"

def format_chat_history(chat_history):
    """Format chat history for inclusion in the prompt"""
    # Skip the most recent message (which is the current query)
    relevant_history = chat_history[:-1] if chat_history else []
    
    if not relevant_history:
        return "No previous conversation."
    
    formatted_history = ""
    for message in relevant_history[-4:]:  # Only include the last 4 messages to keep context manageable
        role = "User" if message["role"] == "user" else "Pregnancy Companion"
        formatted_history += f"{role}: {message['content']}\n\n"
    
    return formatted_history
