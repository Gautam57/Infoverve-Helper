import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import numpy as np

load_dotenv()  # Loads variables from .env into environment

# Retrieve API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment!")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment!")  
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Embedding model setup
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_query",
    google_api_key=GOOGLE_API_KEY
)

# Qdrant setup
client = QdrantClient(host="ai.infoveave.cloud", port=6333)
collections = client.get_collections()
for collection in collections.collections:
    if collection.name == "infoverve_helper_docs":
        collection_name = collection.name
        break

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embedding_model
)

# Query rewriting and semantic fusion
def get_embedding(text):
    return embedding_model.embed_query(text)

def fuse_vectors(vec1, vec2, alpha=0.3):
    fused = alpha * np.array(vec1) + (1 - alpha) * np.array(vec2)
    return fused / np.linalg.norm(fused)

def rewrite_query_with_docs(llm, original_query, docs):
    top_context = "\n\n".join(doc[0].page_content for doc in docs)
    system_prompt = """You are a helpful assistant. 
    Rewrite the user’s query using the relevant documentation. 
    only use the information provided in the documentation to rewrite the query.
    Do not add any additional information or context."""

    prompt = f"""Documentation:{top_context}  Original Query: {original_query} Rewritten Query:"""
    messages = [SystemMessage(content=system_prompt),HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

query = "I need to create a workflow where I initially use the 'Execute API' to get a response and create a data source from it.Then, I want to perform calculations on that data. How do I create this flow?"
# "How do I create a data source in Infoverve?"


# Step 1: Retrieve top docs and original query embedding
docs = vectorstore.similarity_search_with_score(query=query, k=10)
vec_original = get_embedding(query)

# Step 2: Rewrite query using LLM and top docs
llm_rewriter = ChatGroq(model="gemma2-9b-it", temperature=0.01)
rewritten_query = rewrite_query_with_docs(llm_rewriter, query, docs)
vec_rewritten = get_embedding(rewritten_query)

# Step 3: Semantic fusion
vec_fused = fuse_vectors(vec_original, vec_rewritten)

# Step 4: Final retrieval using fused vector
final_docs = client.search(
    collection_name=collection_name,
    query_vector=vec_fused.tolist(),
    limit=10,
    with_payload=True
)

# Step 5: Output
print("🔁 Rewritten Query:", rewritten_query)
for doc in final_docs:
    print(f"[{doc.score:.2f}] {doc.payload.get('page_content', '')}")