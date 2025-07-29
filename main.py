import os
import sys
from dotenv import load_dotenv
import numpy as np
import json

import pandas as pd
from openpyxl import load_workbook

from qdrant_client import QdrantClient
from qdrant_client.http.models import MatchAny, FieldCondition, Filter

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

from scripts.exception import CustomException
from scripts.logger import logging, setup_logger

import streamlit as st

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()  # Loads variables from .env into environment

setup_logger(__file__)

VB_reterived_excel_filepath = "data/results/reterived_docs.xlsx"

# Query rewriting and semantic fusion
def get_embedding(text):
    return embedding_model.embed_query(text)

def fuse_vectors(vec1, vec2, alpha=0.3):
    fused = alpha * np.array(vec1) + (1 - alpha) * np.array(vec2)
    return fused / np.linalg.norm(fused)

def get_infoverve_activities():
    # Load the JSON file
    with open("data/infoverve_content_extractor/infoveave_help_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter all entries with section == "automation"
    automation_entries = [entry for entry in data if entry.get("section") == "automation"]

    # Print or use the content
    activities = []
    for item in automation_entries:
        if "activities" in item['url'].split("/"):
            title = item['content'].split("|")[0]
            activities.append(title)
    return activities


def rewrite_query_with_docs(llm, original_query, docs):
    try:
        activities = get_infoverve_activities()
        # logging.info("Activities related to automation:", activities)

        # Prepare the top context from the documents
        top_context = "\n\n".join(doc[0].page_content for doc in docs)
        with open("data/prompts/rewrittern_query_system_prompt.txt", "r") as file:
            rewritten_query_system_prompt = file.read()
        logging.info("Loaded rewritten query system prompt.")

        rewritten_query_data = {
            "top_context": top_context,
            "original_query": original_query,
            "activities": activities
        }

        with open("data/prompts/rewrittern_query_user_prompt.txt", "r") as file:
            rewritten_query_user_prompt = file.read()
    
        rewritten_query_user_prompt = rewritten_query_user_prompt.format_map(rewritten_query_data)
        logging.info("Loaded rewritten query user prompt.")

        messages = [SystemMessage(content=rewritten_query_system_prompt), HumanMessage(content=rewritten_query_user_prompt)]
        response = llm(messages)
        rewritten = response.content.strip()
    except Exception as e:
        logging.error(f"Error during query rewriting: {str(e)}")
        rewritten = original_query  # fallback to original query if error
    return rewritten

# Retrieve API key (optional: validate it's loaded)
try:
    if "GOOGLE_API_KEY" not in st.session_state:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment!")
        st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        logging.info("GOOGLE_API_KEY loaded into session.")
except Exception as e:
    raise CustomException(f"Failed to load GOOGLE_API_KEY: {str(e)}", sys)

try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment!")
    logging.info("GROQ_API_KEY loaded.")

    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
except Exception as e:
    raise CustomException(f"Failed to load GROQ_API_KEY: {str(e)}", sys)

st.title("Infoverve Helper")
st.write("This application helps you find answers about the Infoverve product using its documentation and Google Generative AI.")
st.caption("Powered by LangChain, Qdrant, and Google Generative AI.")


st.divider()

query = st.text_input("Enter your query:")

if not query:
    st.warning("Please enter a query to proceed.")
else:

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Google's 768-d embedding model
        task_type="retrieval_query",
        google_api_key=st.session_state.GOOGLE_API_KEY
    )
    logging.info("Embedding model initialized.")

    # Connect to Qdrant — update host/port if you're not running locally
    client = QdrantClient(host="ai.infoveave.cloud", port=6333)
    logging.info("Connected to Qdrant.")

    # List all collections
    collections = client.get_collections()
    logging.info("Qdrant collections retrieved.")

    # logging.info collection names
    for collection in collections.collections:
        if collection.name == "infoverve_helper_docs_hybrid":
            collection_name = collection.name
            break
    logging.info(f"Using collection: {collection_name}")

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    logging.info("Sparse embeddings initialized.")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
        
    )
    logging.info("QdrantVectorStore initialized.")

    llm = ChatGroq(
        model="gemma2-9b-it",  # Or other available Groq-hosted open models
        temperature=0.01,
    )
    logging.info("LLM initialized.")

    # Step 1: Retrieve top docs and original query embedding
    initial_docs_with_scores = vectorstore.similarity_search_with_score(query=query, k=10)

    initial_docs = []
    logging.info(initial_docs_with_scores)
    for doc, score in initial_docs_with_scores:
        row = {
            "score": score,
            "page_content": doc.page_content,
        }
        # Add metadata if present
        if doc.metadata:
            row.update(doc.metadata)
        initial_docs.append(row)

    # Create DataFrame
    initial_docs_df = pd.DataFrame(initial_docs)
    initial_docs_df.to_excel(VB_reterived_excel_filepath, sheet_name="initial_docs", index=False)

    vec_original = get_embedding(query)
    logging.info("Original query embedding generated.")

    # Step 2: Rewrite query using LLM and top docs
    rewritten_query = rewrite_query_with_docs(llm, query, initial_docs_with_scores)
    # final_docs_with_vector = []
    logging.info(f"Rewritten query: {rewritten_query}")

    context = []

# Split rewritten query into parts
    MAX_TOTAL_RESULTS = 10
    query_parts = [q.strip() for q in rewritten_query.split("|")]
    logging.info(f"Rewritten query parts: {query_parts}")
    num_parts = len(query_parts)

    # Distribute responses per query, but cap minimum to 1
    no_of_response = max(1, MAX_TOTAL_RESULTS // num_parts)

# Loop through all query parts (even if it's just one)
    for i, q in enumerate(query_parts):
        logging.info(f"Rewritten Query {i+1}: {q}")
        
        vec_rewritten = get_embedding(q)
        logging.info(f"Embedding generated for query {i+1}.")
        
        # vec_fused = fuse_vectors(vec_original, vec_rewritten)
        # logging.info(f"Fused embedding generated for query {i+1}.")
        
        # final_docs_with_vector = vectorstore.similarity_search_by_vector(
        #     embedding=vec_fused.tolist(),
        #     k= no_of_response
        # )
        final_docs_with_score = vectorstore.similarity_search_with_score(query=q, k=no_of_response)

        final_docs = []
        logging.info(final_docs_with_score)
        for doc, score in final_docs_with_score:
            row = {
                "score": score,
                "page_content": doc.page_content,
            }
            # Add metadata if present
            if doc.metadata:
                row.update(doc.metadata)
            final_docs.append(row)
        sheet_name = f"final_docs_{i+1}"
        final_docs_df = pd.DataFrame(final_docs)

        with pd.ExcelWriter(VB_reterived_excel_filepath, engine="openpyxl", mode="a") as writer:
            final_docs_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Add documents to context
        context.extend([
            {
                "page_content": doc[0].page_content,
                "metadata": doc[0].metadata
            }
            for doc in final_docs_with_score
        ])



    # logging.info(f"Rewritten Query: {rewritten_query}")

    # vec_rewritten = get_embedding(rewritten_query)
    # logging.info("Rewritten query embedding generated.")

    # # Step 3: Semantic fusion
    # vec_fused = fuse_vectors(vec_original, vec_rewritten)
    # logging.info("Fused embedding generated.")

    # final_docs_with_vector = vectorstore.similarity_search_by_vector(
    #     embedding=vec_fused.tolist(),
    #     k=5
    # )
    logging.info(f"Found {len(context)} final documents.")

    # source_page_link = []
    # for doc in final_docs_with_vector:
    #     point_id = doc.metadata.get("_id")  # assuming you stored point ID
    #     if point_id:
    #         result = client.retrieve(
    #             collection_name=collection_name,
    #             ids=[point_id],
    #             with_payload=True,
    #         )
    #         page_link = result[0].payload.get('url', '')
    #         source_page_link.append(page_link)

    # source_page_link = list(set(source_page_link))  # Remove duplicates

    # logging.info(f"Retrieved {source_page_link} documents with vectors and payload.")

    # filter = Filter(
    #     must=[
    #         FieldCondition(
    #             key="url",
    #             match=MatchAny(any=source_page_link)  # list of values
    #         )
    #     ]
    # )

    # results, _ = client.scroll(
    #     collection_name=collection_name,
    #     scroll_filter=filter,
    #     with_payload=True,
    #     limit=100,
    # )

    # logging.info(f"Total matches: {len(results)}")
    # logging.info(results)




    # context = [
    #     {
    #         "page_content": doc.page_content,
    #         "metadata": doc.metadata
    #     }
    #     for doc in final_docs_with_vector
    # ]
    logging.info("Context prepared for LLM response.")

    activities = get_infoverve_activities()
    # logging.info("Activities related to automation:", activities)
    main_data = {
        "activities": activities,
    }
    with open("data/prompts/main_system_prompt.txt", "r") as file:
        main_system_prompt = file.read()
    

    main_system_prompt = main_system_prompt.format_map(main_data)
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nUser Query: {query}")
    ]

    logging.info("Generating final answer using LLM...")
    response = llm.invoke(messages)
    logging.info("Final LLM Response:\n")
    logging.info(response.content)

    # Save response to a Markdown file
    output_md_path = "./data/results/infoverve_helper_response.md"
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(response.content)

    logging.info(f"LLM response saved to {output_md_path}")
    st.markdown(response.content)

