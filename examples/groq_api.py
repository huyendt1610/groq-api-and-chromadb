# !pip install -q groq python-dotenv requests beautifulsoup4
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

def stream_chat_completion(client):
    streams = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful asssistant. Answer as John Snow'
            },
            {
                'role': 'user',
                'content': 'Explain the importance of low latency LLMs'
            }
        ],
        model='llama-3.1-8b-instant',
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True
    )

    for chunk in streams:
        print(chunk.choices[0].delta.content, end='')

def non_stream_chat_completion(client):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful asssistant. Answer as John Snow'
            },
            {
                'role': 'user',
                'content': 'Explain the importance of low latency LLMs'
            }
        ],
        model='llama-3.1-8b-instant',
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False
    )
    print(chat_completion.choices[0].message.content)

def stream_chat_completion_with_web_scraping(client):
    from bs4 import BeautifulSoup
    import requests

    url = "https://paulgraham.com/greatwork.html"

    response = requests.get(url)
    html = response.text

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    print(f"Text length: {len(text)} chars, ~{len(text)//4} tokens")

    max_chars = 20000  # ~5000 tokens, safe under 12k TPM limit

    response = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant. Your job is to summarize the following text in 10 points'},
            {'role': 'user', 'content': text[:max_chars]}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True
    )

    for stream_chunk in response:
        print(stream_chunk.choices[0].delta.content or '', end="")


def main():
    api_key = os.getenv('GROQ_API_KEY')
    client = Groq(api_key=api_key)

    # non_stream_chat_completion(client)
    # stream_chat_completion(client)
    stream_chat_completion_with_web_scraping(client)


if __name__ == "__main__":
    main()
