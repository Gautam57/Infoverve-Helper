import os
import sys
import json
from uuid import uuid4
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct,SparseVectorParams
from rank_bm25 import BM25Okapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Adjust path as needed
from scripts.exception import CustomException
from scripts.logger import logging, setup_logger

setup_logger(__file__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment!")
logging.info("GOOGLE_API_KEY loaded.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Config ---
JSON_FILE = "data/infoverve_content_extractor/infoveave_help_data.json"
COLLECTION_NAME = "infoverve_helper_eachpage_hybrid"
QDRANT_HOST = "ai.infoveave.cloud"
QDRANT_PORT = 6333
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 100


# --- Load Data ---
with open(JSON_FILE, "r", encoding="utf-8") as f:
    all_pages = json.load(f)
logging.info(f"Loaded {len(all_pages)} pages.")

def count_words(text):
    return len(text.split())

# --- Initialize Embedding Model ---
logging.info("Initializing embedding model...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document",
    google_api_key=GOOGLE_API_KEY
)
logging.info("Embedding model initialized.")

# --- Initialize Qdrant ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
logging.info("Connected to Qdrant.")

# Create collection if not exists
if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)})
logging.info(f"Collection '{COLLECTION_NAME}' is ready.")

# --- Process and Upload ---
all_points = []

logging.info(f"Processing {len(all_pages)} pages...")

for page in tqdm(all_pages):
    logging.info(f"Processing page: {page.get('Page_title', 'Unknown')}")

    chunks = page.get("content", "")
    embeddings = embedding_model.embed_documents(chunks)
    logging.info(f"Generated embeddings for page: {page.get('Page_title', 'Unknown')}")
    payload = {
            "url": page.get("url"),
            "title": page.get("Page_title"),
            "section": page.get("section"),
            "terminologies": page.get("Terminologies", []),
            "char_count": page.get("no_of_char", 0),
            "word_count": page.get("no_of_words", 0),
            "page_content": chunks
        }
    logging.info(f"Payload prepared for page: {page.get('Page_title', 'Unknown')}")
        # sparse_vec = sparse_vectorizer(chunk)
    point = PointStruct(
        id=str(uuid4()),
        vector={"dense": embeddings},
        payload=payload
    )
    all_points.append(point)
    logging.info(f"Point created for page: {page.get('Page_title', 'Unknown')}")    

# Upload in batches
logging.info(f"Uploading {len(all_points)} vectors in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(all_points), BATCH_SIZE)):
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=all_points[i:i + BATCH_SIZE]
        )
    except Exception as e:
        logging.error(f"Error uploading batch {i // BATCH_SIZE + 1}: {e}")
        raise CustomException(f"Failed to upload batch {i // BATCH_SIZE + 1}: {e}")
    

logging.info("✅ Upload complete.")
