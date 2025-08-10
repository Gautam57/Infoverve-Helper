import json
import sys
from uuid import uuid4
from tqdm import tqdm
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking_evaluation.chunking import ClusterSemanticChunker

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient,models
from qdrant_client.models import VectorParams, Distance, PointStruct,SparseVectorParams
from rank_bm25 import BM25Okapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
COLLECTION_NAME = "infoverve_docs_kg_hybrid"
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
# --- Initialize Text Splitter ---
# Using ClusterSemanticChunker for semantic chunking
logging.info("Initializing text splitter...")
text_splitter = ClusterSemanticChunker(
    embedding_function=embedding_model.embed_documents,
    max_chunk_size=CHUNK_SIZE,       # tokens
    length_function=count_words       # or a custom token counter
)
logging.info("Text splitter initialized.")

# --- Initialize Qdrant ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
logging.info("Connected to Qdrant.")

# Create collection if not exists
if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))})
logging.info(f"Collection '{COLLECTION_NAME}' is ready.")

# --- Fit TF-IDF on all chunks ---
all_chunks = []
for page in tqdm(all_pages):
    all_chunks.extend(text_splitter.split_text(page.get("content", "")))
logging.info(f"Total chunks for TF-IDF: {len(all_chunks)}")

# tfidf = TfidfVectorizer()
# tfidf.fit(all_chunks)
# logging.info("TF-IDF model fitted.")

# def sparse_vectorizer(text):
#     vec = tfidf.transform([text])
#     indices = vec.indices.tolist()
#     values = vec.data.tolist()
#     return {"indices": indices, "values": values}

# Tokenize the corpus
tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_chunks)
logging.info("BM25 model fitted.")

# # Build vocabulary index for sparse representation
# vocab = {word: idx for idx, word in enumerate(set(word for doc in tokenized_chunks for word in doc))}

def sparse_vectorizer(text: str) -> dict:
    tokens = text.lower().split()
    scores = bm25.get_scores(tokens)

    # Get non-zero BM25 scores and convert to sparse-like format
    values = []
    indices = []
    for i, score in enumerate(scores):
        if score > 0:
            indices.append(i)
            values.append(score)

    return {"indices": indices, "values": values}


# --- Process and Upload ---
all_points = []

logging.info(f"Processing {len(all_pages)} pages...")

for page in tqdm(all_pages):

    chunks = text_splitter.split_text(page.get("content", ""))
    embeddings = embedding_model.embed_documents(chunks)
    # logging.info(f"Page '{page.get('Page_title', 'Unknown')}' has {len(chunks)} chunks.")

    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        payload = {
            "url": page.get("url"),
            "title": page.get("Page_title"),
            "section": page.get("section"),
            "terminologies": page.get("Terminologies", []),
            "char_count": page.get("no_of_char", 0),
            "word_count": page.get("no_of_words", 0),
            "page_content": chunk,
            "chunk_index": idx
        }

        payload["entities"] = ([{"name": term, "type": "Concept"} for term in page.get("Terminologies", [])] 
        + [{"name": page.get("Page_title"), "type": "Widget"},{"name": page.get("section"), "type": "Section"},])

        payload["triplets"] = ([(page.get("Page_title"), "belongs_to", page.get("section"))] 
        + [ (page.get("Page_title"), "relates_to", term) for term in page.get("Terminologies", [])])


        sparse_vec = sparse_vectorizer(chunk)
        point = PointStruct(
            id=str(uuid4()),
            vector={"dense": vector,
                    "sparse": models.SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"])},
            payload=payload
        )
        all_points.append(point)

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
