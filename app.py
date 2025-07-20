import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document



load_dotenv()  # Loads variables from .env into environment

# Retrieve API key (optional: validate it's loaded)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Google's 768-d embedding model
    task_type="retrieval_query",
    google_api_key=GOOGLE_API_KEY
)

query = "steps to create a data source?"
query_vector = embedding_model.embed_query(query)

# Connect to Qdrant — update host/port if you're not running locally
client = QdrantClient(host="ai.infoveave.cloud", port=6333)

# List all collections
collections = client.get_collections()

# Print collection names
for collection in collections.collections:
    if collection.name == "infoverve_helper_collection":
        print(f"Collection Name: {collection.name}")
        collection_name = collection.name
        break

results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5,
    with_payload=True
)

docs = [
    {
        "document": Document(
            page_content=record.payload.get("content", ""),
            metadata={
                "id": record.id,
                "title": record.payload.get("title", ""),
                "url": record.payload.get("url", ""),
                "section": record.payload.get("section", ""),
                "terminologies": record.payload.get("terminologies", []),
                "char_count": record.payload.get("char_count", 0),
                "word_count": record.payload.get("word_count", 0),
                "chunk_index": record.payload.get("chunk_index", 0)
            }
        ),
        "score": record.score
    }
    for record in results
]

print(f"Found {len(docs)} documents matching the query.")
for item in docs:
    if item["score"] < 0.1:
        continue
    else:
        doc = item["document"]
        score = item["score"]
        print(f"Score: {score:.4f}")
        print(doc.page_content)
        print(doc.metadata)
        print("-----")


# # Connect to existing collection
# vectorstore = QdrantVectorStore(
#     client=client,
#     collection_name=collection_name,  # Replace with your collection name
#     embedding=embedding_model,
# )

# query = "How to create a Data Source in infoverve?"

# results = vectorstore.similarity_search(query, k=5)

# for doc in results:
#     print(doc.page_content)
