# Groq API Demo

A Python project demonstrating how to use the [Groq API](https://groq.com/) for fast LLM inference, including streaming completions, web scraping, RAG pipelines, and Streamlit chat apps.

## Features

- **Streaming chat completion** — streams tokens in real time using `llama-3.1-8b-instant`
- **Non-streaming chat completion** — returns the full response at once
- **Web scraping + summarization** — scrapes a webpage with BeautifulSoup and summarizes it using `llama-3.3-70b-versatile`
- **Vector search with ChromaDB** — embeds documents from a text file and runs semantic similarity queries
- **Groq chat app** — Streamlit chat interface with conversation memory and model selection
- **OpenAI chat app** — Streamlit RAG app that ingests PDFs, builds a FAISS vector index, and answers questions using GPT-4o-mini
- **RAG app (main)** — Same RAG pipeline with a sidebar UI for uploading and managing PDF documents

## Project Structure

```
groq-api/
├── groq_api.py              # Streaming, non-streaming, and web-scraping demos
├── chroma_db.py             # ChromaDB vector search demo
├── apps/
│   ├── groq_chat/app.py     # Streamlit chat app powered by Groq + LangChain
│   ├── openai_chat/app.py   # RAG app (OpenAI + FAISS, no upload UI)
│   └── rag/app.py           # RAG app with PDF upload sidebar
├── notebooks/
│   ├── ai_agent.ipynb       # AI agent notebook
│   └── check_semantic_search.ipynb
├── data/
│   └── papers/              # PDF files for RAG indexing
└── vector_index/            # Persisted FAISS index
```

## Requirements

- Python 3.8+
- A [Groq API key](https://console.groq.com/)
- An [OpenAI API key](https://platform.openai.com/) (for RAG apps)

## Setup

1. Clone the repo and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Groq API (streaming & web scraping)

```bash
python groq_api.py
```

By default runs `stream_chat_completionwith_web_scraping`, which fetches Paul Graham's *Great Work* essay and summarizes it in 10 points. To try other modes, uncomment the relevant lines in `main()`:

```python
def main():
    # non_stream_chat_completion(client)
    # stream_chat_completion(client)
    stream_chat_completionwith_web_scraping(client)
```

### ChromaDB Vector Search

```bash
python chroma_db.py
```

Reads lines from `data/policies.txt`, adds them to an in-memory ChromaDB collection with auto-generated embeddings, then queries the collection with semantic search questions.

### Groq Chat App

```bash
streamlit run apps/groq_chat/app.py
```

A Streamlit chat interface powered by LangChain + Groq. Features:
- Model selection (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-120b`)
- Configurable conversation memory length (1–10 turns)

### OpenAI Chat App — RAG with FAISS

```bash
streamlit run apps/openai_chat/app.py
```

A retrieval-augmented generation (RAG) app that:
1. Loads PDF files from `./data/papers/`
2. Splits and embeds them with `text-embedding-3-small`
3. Stores/loads a FAISS vector index locally (`vector_index/`)
4. Answers questions using `gpt-4o-mini` with citations

### RAG App (with PDF upload UI)

```bash
streamlit run apps/rag/app.py
```

Same RAG pipeline as above, with an added sidebar for uploading PDF files directly from the browser, viewing indexed files, and deleting them.

## Models Used

| Script / App | Model |
|---|---|
| `groq_api.py` — `stream_chat_completion` | `llama-3.1-8b-instant` |
| `groq_api.py` — `non_stream_chat_completion` | `llama-3.1-8b-instant` |
| `groq_api.py` — `stream_chat_completionwith_web_scraping` | `llama-3.3-70b-versatile` |
| `apps/groq_chat/app.py` | selectable (default: `llama-3.3-70b-versatile`) |
| `apps/openai_chat/app.py` / `apps/rag/app.py` — embeddings | `text-embedding-3-small` |
| `apps/openai_chat/app.py` / `apps/rag/app.py` — QA | `gpt-4o-mini` |
