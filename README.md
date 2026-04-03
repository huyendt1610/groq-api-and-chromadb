# Groq API Demo

A Python project demonstrating how to use the [Groq API](https://groq.com/) for fast LLM inference, including streaming completions and web scraping integration.

## Features

- **Streaming chat completion** — streams tokens in real time using `llama-3.1-8b-instant`
- **Non-streaming chat completion** — returns the full response at once
- **Web scraping + summarization** — scrapes a webpage with BeautifulSoup and summarizes it using `llama-3.3-70b-versatile`
- **Vector search with ChromaDB** — embeds documents from a text file and runs semantic similarity queries

## Requirements

- Python 3.8+
- A [Groq API key](https://console.groq.com/)

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
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Groq API

```bash
python groq_api.py
```

By default it runs `stream_chat_completionwith_web_scraping`, which fetches Paul Graham's *Great Work* essay and summarizes it in 10 points. To try the other modes, uncomment the relevant lines in `main()`:

```python
def main():
    ...
    # non_stream_chat_completion(client)
    # stream_chat_completion(client)
    stream_chat_completionwith_web_scraping(client)
```

### ChromaDB Vector Search

```bash
python chroma_db.py
```

Reads lines from `policies.txt`, adds them to an in-memory ChromaDB collection with auto-generated embeddings, then queries the collection with semantic search questions. Output shows the top matching documents for each query.

## Models Used

| Function | Model |
|---|---|
| `stream_chat_completion` | `llama-3.1-8b-instant` |
| `non_stream_chat_completion` | `llama-3.1-8b-instant` |
| `stream_chat_completionwith_web_scraping` | `llama-3.3-70b-versatile` |
