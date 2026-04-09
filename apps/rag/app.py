import streamlit as st
import os
import glob
import gc
import pathlib
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import time
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
    loader = DirectoryLoader(str(DATA_DIR), glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader, use_multithreading=False)
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


st.title("Main App")
st.sidebar.title("Navigation")

uploaded_files = st.sidebar.file_uploader("Choose PDF files", key="file1", type=["pdf"], accept_multiple_files=True)
process_clicked = st.sidebar.button("Upload & Build Index")
main_placeholder = st.empty()

# Show existing files in sidebar
existing_files = glob.glob(str(DATA_DIR / "**" / "*.pdf"), recursive=True)
if existing_files:
    st.sidebar.markdown("**Indexed files:**")
    for f in existing_files:
        col1, col2 = st.sidebar.columns([4, 1])
        col1.text(os.path.basename(f))
        if col2.button("🗑", key=f"del_{f}"):
            gc.collect()
            os.remove(f)
            st.rerun()

if process_clicked:
    if not uploaded_files:
        st.sidebar.warning("Please select at least one PDF file first.")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            save_path = DATA_DIR / file.name
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
                st.sidebar.text(f"Saved {file.name}")

        with st.sidebar.status("Building index...", expanded=True) as status:
            build_vector_index(VECTOR_INDEX_PATH)
            status.update(label="Index built!", state="complete")
        del st.session_state["file1"]
        st.rerun()

if index_is_stale(VECTOR_INDEX_PATH):
    main_placeholder.text("Index is stale or missing, rebuilding...")
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

user_question = st.chat_input("Ask a question to the OpenAI model:")
if user_question:
    results = retriever.invoke(user_question)
    print(f"Retrieved {len(results)} chunks")
    answer = rag_chain.invoke(user_question)
    print(f"Question: {user_question}\nAnswer: {answer}")
    st.markdown("**Answer:**")
    st.write(answer)
