import json
from uuid import uuid4
from tqdm import tqdm
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Config ---
JSON_FILE = "/Users/gautambr/Documents/Infoverve Helper/infoverve_content_extractor/data/infoveave_help_data.json"
COLLECTION_NAME = "infoverve_helper_collection"
QDRANT_HOST = "ai.infoveave.cloud"
QDRANT_PORT = 6333
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
BATCH_SIZE = 100

# --- Load Data ---
with open(JSON_FILE, "r", encoding="utf-8") as f:
    all_pages = json.load(f)

# --- Initialize Embedding Model ---
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    google_api_key=GOOGLE_API_KEY
)

# --- Initialize Chunker ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

# --- Initialize Qdrant ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Create collection if not exists
if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# --- Process and Upload ---
all_points = []

print(f"Processing {len(all_pages)} pages...")

for page in tqdm(all_pages):
    metadata = {
        "url": page.get("url"),
        "title": page.get("Page_title"),
        "section": page.get("section"),
        "terminologies": page.get("Terminologies", []),
        "char_count": page.get("no_of_char", 0),
        "word_count": page.get("no_of_words", 0),
    }

    chunks = text_splitter.split_text(page.get("content", ""))
    embeddings = embedding_model.embed_documents(chunks)

    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        payload = metadata.copy()
        payload["content"] = chunk
        payload["chunk_index"] = idx

        point = PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload=payload
        )
        all_points.append(point)

# Upload in batches
print(f"Uploading {len(all_points)} vectors in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(all_points), BATCH_SIZE)):
    client.upsert(collection_name=COLLECTION_NAME, points=all_points[i:i + BATCH_SIZE])

print("✅ Upload complete.")
