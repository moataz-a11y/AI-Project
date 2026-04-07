import os
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


def get_anthropic_api_key() -> Optional[str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"].strip()
    try:
        return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except (FileNotFoundError, KeyError, TypeError):
        return None


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
    st.caption(
        "Answers use **Claude** (Anthropic). Search uses a **local** embedding model "
        "(no extra API key; first run downloads ~80MB)."
    )

api_key = get_anthropic_api_key()
if not api_key:
    st.warning(
        "Set your Claude API key: `export ANTHROPIC_API_KEY='...'` before running, "
        "or add `ANTHROPIC_API_KEY = \"...\"` to `.streamlit/secrets.toml`."
    )

if file is not None and api_key:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            text += (page_text or "") + "\n"

    text = text.strip()
    if not text:
        st.error("No text could be extracted from this PDF. Try another file or a text-based PDF.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            st.error("Could not split the document into chunks.")
        else:
            with st.spinner("Indexing document (embeddings)…"):
                vector_store = FAISS.from_texts(chunks, get_embeddings())

            user_question = st.text_input("Type your question here")

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4},
            )

            claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip()
            llm = ChatAnthropic(
                model=claude_model,
                temperature=0.3,
                max_tokens=1000,
                api_key=api_key,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant answering questions about a PDF document.\n\n"
                        "Guidelines:\n"
                        "1. Provide complete, well-explained answers using the context below.\n"
                        "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
                        "3. If the context mentions related information, include it to give fuller picture.\n"
                        "4. Only use information from the provided context - do not use outside knowledge.\n"
                        "5. Summarize long information, ideally in bullets where needed\n"
                        "6. If the information is not in the context, say so politely.\n\n"
                        "Context:\n{context}",
                    ),
                    ("human", "{question}"),
                ]
            )

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            if user_question:
                with st.spinner("Thinking…"):
                    response = chain.invoke(user_question)
                st.write(response)
