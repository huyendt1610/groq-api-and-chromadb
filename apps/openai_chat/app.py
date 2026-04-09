import os
import glob
import pathlib
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, DirectoryLoader, UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "papers"
VECTOR_INDEX_PATH = str(ROOT / "vector_index")


def build_vector_index(filepath):
    loader = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader, use_multithreading=True)
    data = loader.load()
    print(f"Number of documents loaded: {len(data)}")

    MARKDOWN_SEPARATORs = [
        "\n#{1,6} ",
        "```n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        ""
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=MARKDOWN_SEPARATORs)
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorindex = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
    vectorindex.save_local(filepath)

def load_vector_index(filepath):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorindex = FAISS.load_local(filepath, embeddings, allow_dangerous_deserialization=True)
    return vectorindex

def index_is_stale(index_path, docs_path=None):
    if docs_path is None:
        docs_path = str(DATA_DIR)
    if not os.path.exists(index_path):
        return True
    index_mtime = os.path.getmtime(index_path)
    pdf_files = glob.glob(f"{docs_path}/**/*.pdf", recursive=True)
    return any(os.path.getmtime(f) > index_mtime for f in pdf_files)

def main():
    main_placeholder = st.empty()

    if index_is_stale(VECTOR_INDEX_PATH):
        print("Index is stale or missing, rebuilding...")
        build_vector_index(VECTOR_INDEX_PATH)

    vectorindex = load_vector_index(VECTOR_INDEX_PATH)
    print("Vector index loaded from disk.")

    retriever = vectorindex.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1. Only use information from the provided context.\n"
        "2. If you don't know the answer, say so.\n"
        "3. Cite your sources if applicable using the metadata.\n"
        "Context:\n{context}\n\n"
        "Question:{question}\n"
    )

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    rag_chain = (
        { "context": retriever, "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    user_question = main_placeholder.text_input("Ask a question to the OpenAI model:")
    if user_question:
        results = retriever.invoke(user_question)
        print(f"Retrieved {len(results)} chunks")
        answer = rag_chain.invoke(user_question)
        print(f"Question: {user_question}\nAnswer: {answer}")

if __name__ == "__main__":
    main()
