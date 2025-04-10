import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_documents(uploaded_files: list[UploadedFile]) -> list[Document]:
    """Processes multiple uploaded documents (pdf, txt, docx, md, etc.)."""
    all_docs = []

    for uploaded_file in uploaded_files:
        file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix).name

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_suffix == ".pdf":
            loader = PyMuPDFLoader(temp_file_path)
        elif file_suffix == ".txt":
            loader = TextLoader(temp_file_path)
        elif file_suffix == ".md":
            loader = UnstructuredMarkdownLoader(temp_file_path)
        elif file_suffix == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        else:
            loader = UnstructuredFileLoader(temp_file_path)

        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    return text_splitter.split_documents(all_docs)


def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success(f" Data from `{file_name}` added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    if not documents:
        return "", []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    relevant_text = ""
    relevant_text_ids = []

    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]] + "\n\n"
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text.strip(), relevant_text_ids


# Streamlit UI
st.set_page_config(page_title="RAG Question Answer", layout="wide")

with st.sidebar:
    st.title("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "**Upload documents for QnA**", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
    )
    process = st.button("⚡️ Process & Embed")

    if uploaded_files and process:
        all_splits = process_documents(uploaded_files)
        for uploaded_file in uploaded_files:
            normalized_name = uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
            add_to_vector_collection(all_splits, normalized_name)

st.header("🗣️ Ask a Question Based on Your Files")
prompt = st.text_area("**Ask a question:**")
ask = st.button("🔥 Get Answer")

if ask and prompt:
    results = query_collection(prompt)
    context = results.get("documents")[0]
    relevant_text, relevant_ids = re_rank_cross_encoders(context, prompt)
    response = call_llm(context=relevant_text, prompt=prompt)
    st.write_stream(response)

    with st.expander(" Retrieved Chunks"):
        st.write(results)

    with st.expander(" Top Ranked Context (IDs)"):
        st.write(relevant_ids)
        st.write(relevant_text)
