import chromadb 
import uuid

client = chromadb.Client() # PersistentClient(path="./chroma_data") # default is in-memory, but you can specify a directory for persistence


collection = client.get_or_create_collection(name="my_collection") # DefaultEmbeddingFunction

with open("policies.txt", "r", encoding="utf-8") as f:
    policies: list[str] = f.read().splitlines()


collection.add(
    ids=[str(uuid.uuid4()) for _ in policies],
    documents=policies,
    metadatas=[{"line_number": i} for i in range(len(policies))],
)

print(collection.peek(5)) # return both uuids, embeddings, documents, and metadatas for the first 5 entries in the collection

# it embedded this query => ran similarity search in embeddings in our index 
results = collection.query(
    query_texts=[
        "How do I set up a virtual environment for Chroma?",
        "How do I authenticate with Chroma Cloud?",
        "How do I create and connect to a Chroma database?",
    ],
    n_results=5, # default is 10, number of results to return for each query
)

for i, query_results in enumerate(results["documents"]):
    print(f"\nQuery: {i}")
    print("\n".join(query_results))
    