import os
import requests
import random
from typing import Optional

import pdfplumber
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG & MIRO HELPERS ---
MIRO_AUTH_URL = "https://miro.com/oauth/authorize"
MIRO_TOKEN_URL = "https://api.miro.com/v1/oauth/token"
# Make sure this matches exactly what you put in the Miro Developer Console
REDIRECT_URI = "http://localhost:8501/"if st.get_option("server.address") == "localhost" else "https://ai-project-xko35uwkou6asd6hfucnvv.streamlit.app/"

def get_anthropic_api_key() -> Optional[str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"].strip()
    try:
        return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except (FileNotFoundError, KeyError, TypeError):
        return None

def exchange_code_for_token(auth_code: str):
    """Exchanges Miro auth code for an access token."""
    data = {
        "grant_type": "authorization_code",
        "client_id": st.secrets["MIRO_CLIENT_ID"],
        "client_secret": st.secrets["MIRO_CLIENT_SECRET"],
        "code": auth_code,
        "redirect_uri": REDIRECT_URI
    }
    response = requests.post(MIRO_TOKEN_URL, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    return None

def push_to_miro_doc(token: str, board_id: str, content: str, question: str):
    """Creates a Miro Doc using the correct v2 REST parameters."""
    url = f"https://api.miro.com/v2/boards/{board_id}/docs"
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {token}"
    }
    
    # Randomize position
    random_x = random.randint(-1000, 1000)
    random_y = random.randint(-1000, 1000)
    
    # CLEANED PAYLOAD: Title is moved INSIDE the content string
    payload = {
        "data": {
            "contentType": "markdown",
            "content": f"# Analysis: {question}\n\n{content}" # Title goes here!
        },
        "position": {
            "x": random_x,
            "y": random_y,
            "origin": "center"
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 201:
        # This will help you see if there are other parameters Miro dislikes
        st.error(f"Miro Error {response.status_code}: {response.text}")
        
    return response.status_code == 201

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# --- APP UI ---
# DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
st.header("AI PDF Analyst + Miro")

# --- MIRO OAUTH LOGIC ---
# Detect if user is returning from Miro with an auth code
if "code" in st.query_params and "miro_token" not in st.session_state:
    with st.spinner("Finalizing Miro connection..."):
        token = exchange_code_for_token(st.query_params["code"])
        if token:
            st.session_state.miro_token = token
            st.toast("Connected to Miro!")
            # Clear the code from URL for a clean look
            st.query_params.clear()

with st.sidebar:
    st.title("Settings")
    
    # Miro Section
    st.subheader("Miro Integration")
    if "miro_token" not in st.session_state:
        auth_link = f"{MIRO_AUTH_URL}?response_type=code&client_id={st.secrets['MIRO_CLIENT_ID']}&redirect_uri={REDIRECT_URI}"
        st.link_button("🔐 Connect Miro Account", auth_link)
    else:
        st.success("✅ Miro Connected")
        st.session_state.board_id = st.text_input("Board ID", placeholder="Paste Board ID here...")
        if st.button("Log out of Miro"):
            del st.session_state.miro_token
            st.rerun()

    st.divider()
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")
    st.caption("Answers use **Claude 4.6**. Search uses a **local** embedding model.")

# --- CORE LOGIC ---
api_key = get_anthropic_api_key()

if not api_key:
    st.warning("Please set your ANTHROPIC_API_KEY in secrets.toml.")

if file is not None and api_key:
    # PDF Processing (Existing logic)
    with pdfplumber.open(file) as pdf:
        text = "".join([(page.extract_text() or "") + "\n" for page in pdf.pages]).strip()

    if not text:
        st.error("No text found in PDF.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        with st.spinner("Indexing document..."):
            vector_store = FAISS.from_texts(chunks, get_embeddings())
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

        user_question = st.text_input("Ask a question about the document:")

        if user_question:
            llm = ChatAnthropic(model=DEFAULT_CLAUDE_MODEL, temperature=0.3, api_key=api_key)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Context: {context}"),
                ("human", "{question}")
            ])
            chain = ({"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                      "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

            with st.spinner("Thinking..."):
                response = chain.invoke(user_question)
            
            st.markdown(f"### Answer\n{response}")

            # --- MIRO PUSH BUTTON (ROBUST VERSION) ---
if "miro_token" in st.session_state:
    # Scenario A: Logged in and has a Board ID
    if st.session_state.get("board_id"):
        if st.button("📄 Push to Miro as Doc"):
            with st.spinner("Creating Miro Doc..."):
                # We pass the token, cleaned board_id, the AI response, and the original question
                success = push_to_miro_doc(
                    st.session_state.miro_token, 
                    st.session_state.board_id, 
                    response,
                    user_question
                )
                
                if success:
                    st.success("Analysis Doc created on Miro!")
                else:
                    st.error("Failed to push. Check your Board ID and Permissions in Miro Console.")
    
    # Scenario B: Logged in but FORGOT the Board ID
    else:
        st.info("💡 Paste a Board ID in the sidebar to push this analysis to Miro.")
        
else:
    # Scenario C: Not logged in at all
    st.warning("🔐 Connect your Miro account in the sidebar to export this to a board.")