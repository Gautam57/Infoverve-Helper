from groq import Groq
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Qdrant
import os
from dotenv import load_dotenv


load_dotenv()  # Loads variables from .env into environment

# Retrieve API key (optional: validate it's loaded)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Google's 768-d embedding model
    task_type="retrieval_query"
)


# Connect to Qdrant — update host/port if you're not running locally
client = QdrantClient(host="ai.infoveave.cloud", port=6333)

# List all collections
collections = client.get_collections()

# Print collection names
for collection in collections.collections:
    collection_name = collection.name
    print(f"Collection Name: {collection_name}")


# Connect to existing collection
vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,  # Replace with your collection name
    embeddings=embedding_model,
)

query = "How to create a Data Source in infoverve?"

results = vectorstore.similarity_search(query, k=5)

for doc in results:
    print(doc.page_content)
