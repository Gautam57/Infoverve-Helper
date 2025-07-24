import json
import os
from uuid import uuid4
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunking_evaluation.chunking import ClusterSemanticChunker
from langchain.schema import Document
from langchain.vectorstores import Qdrant

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Config ---
JSON_FILE = "infoverve_content_extractor/data/infoveave_help_data.json"
COLLECTION_NAME = "infoverve_helper_docs"
QDRANT_HOST = "ai.infoveave.cloud"
QDRANT_PORT = 6333
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

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
def count_words(text):
    return len(text.split())
text_splitter = ClusterSemanticChunker(
    embedding_function=embedding_model.embed_documents,
    max_chunk_size=500,       # tokens
    length_function=count_words       # or a custom token counter
)

# --- Prepare Documents ---
documents = []
for page in tqdm(all_pages, desc="Chunking pages"):
    metadata = {
        "url": page.get("url"),
        "title": page.get("Page_title"),
        "section": page.get("section"),
        "terminologies": page.get("Terminologies", []),
        "char_count": page.get("no_of_char", 0),
        "word_count": page.get("no_of_words", 0),
    }
    chunks = text_splitter.split_text(page.get("content", ""))
    for idx, chunk in enumerate(chunks):
        doc_metadata = metadata.copy()
        doc_metadata["chunk_index"] = idx
        documents.append(Document(page_content=chunk, metadata=doc_metadata))

print(f"Prepared {len(documents)} document chunks.")

# --- Upload to Qdrant using LangChain ---
vectorstore = Qdrant.from_documents(
    documents,
    embedding_model,
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    collection_name=COLLECTION_NAME,
    force_recreate=False
)

print("✅ Upload complete.")