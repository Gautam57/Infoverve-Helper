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
from neo4j import GraphDatabase

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

load_dotenv()  # Loads variables from .env into environment

setup_logger(__file__)

VB_reterived_excel_filepath = "data/results/reterived_docs.xlsx"

NEO4J_URI = "neo4j+s://034f8de5.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

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

def llmcontextBuilder(docs,collection_name, client):
    All_context_with_MD = []
    logging.info("Building context with metadata from documents...")
    for doc in docs:
        # logging.info(doc)
        point_id = doc[0].metadata.get("_id")  # assuming you stored point ID
        if point_id:
            result = client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )
            logging.info(result)
            logging.info(f"Retrieved result for point ID: {point_id}")
            context_with_metadata = {"page_content": result[0].payload.get("page_content", ""),
                                        "url": result[0].payload.get("url", ""),
                                        "title": result[0].payload.get("title", ""),
                                        "section": result[0].payload.get("section", ""),
                                        "terminologies": result[0].payload.get("terminologies", []),
                                        # "char_count": result[0].payload.get("char_count", 0),
                                        # "word_count": result[0].payload.get("word_count", 0),
                                        # "chunk_index": result[0].payload.get("chunk_index", None),
                                        # "entities": result[0].payload.get("entities", []),
                                        # "triplets": result[0].payload.get("triplets", []),
                                        "id": point_id
                                        }
            # print(context_with_metadata)

            All_context_with_MD.append(context_with_metadata)
    # logging.info(All_context_with_MD)
    return All_context_with_MD

def entities_retrivel(docs,collection_name, client):
    entities = []
    logging.info("Building context with metadata from documents...")
    for doc in docs:
        # logging.info(doc)
        point_id = doc[0].metadata.get("_id")  # assuming you stored point ID
        if point_id:
            result = client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )
            logging.info(result)
            logging.info(f"Retrieved result for point ID: {point_id}")
            entities_triplets = {
                "entities": result[0].payload.get("entities", []),
                "triplets": result[0].payload.get("triplets", []),
                }
            # print(context_with_metadata)

            entities.append(entities_triplets)
    # logging.info(All_context_with_MD)
    return entities

def query_neo4j_for_entities(entities, hops=1, top_n=20):
    # returns list of (head, rel, tail) from Neo4j related to given entities
    logging.info(f"Querying Neo4j for entities: {entities} with hops={hops} and top_n={top_n}")
    names = ["Concept", "Widget", "Section"]

    query = """
            CALL {
                    UNWIND $entities AS ent
                    MATCH (e:`Concept` {name: ent[0]})-[r]-(m)
                    WHERE ent[1] = 'Concept'
                    RETURN e.name AS head, type(r) AS rel, m.name AS tail, COUNT(*) AS freq
                    UNION
                    UNWIND $entities AS ent
                    MATCH (e:`Widget` {name: ent[0]})-[r]-(m)
                    WHERE ent[1] = 'Widget'
                    RETURN e.name AS head, type(r) AS rel, m.name AS tail, COUNT(*) AS freq
                    UNION
                    UNWIND $entities AS ent
                    MATCH (e:`Section` {name: ent[0]})-[r]-(m)
                    WHERE ent[1] = 'Section'
                    RETURN e.name AS head, type(r) AS rel, m.name AS tail, COUNT(*) AS freq
                }
                RETURN head, rel, tail, freq
                ORDER BY freq DESC
            LIMIT $limit

            """

    with driver.session() as session:
        res = session.run(query, entities=entities, limit=20)
        triplets = [(row["head"], row["rel"], row["tail"]) for row in res]

    # print(triplets)
    logging.info(triplets)
    logging.info(f"Retrieved {len(triplets)} triplets from Neo4j.")
    return triplets

def rewrite_query_with_docs(llm, original_query, docs):
    try:
        activities = get_infoverve_activities()
        # logging.info("Activities related to automation:", activities)
        top_context = llmcontextBuilder(docs, collection_name, client)
        logging.info("Top context prepared for query rewriting.")
        retrived_entities = entities_retrivel(docs, collection_name, client)
        entities = set()
        for h in retrived_entities:
            # Handle entities that already have name & type
            for ent in h.get("entities", []):
                if isinstance(ent, dict):
                    entities.add((ent.get("name", ""), ent.get("type", "")))
          
        # Query Neo4j for related facts
        neo_triplets = query_neo4j_for_entities(list(entities), top_n=20)

        # logging.info(f"Retrieved {len(neo_triplets)} related triplets from Neo4j.")

        kg_lines = [f"{h} —[{r}]→ {t}" for (h, r, t) in neo_triplets]
        kg_text = "\n".join(kg_lines)
        
        logging.info("Knowledge graph facts prepared for query rewriting.")
        # Prepare the top context from the documents
        
        with open("data/prompts/rewrittern_query_system_prompt.txt", "r") as file:
            rewritten_query_system_prompt = file.read()
        logging.info("Loaded rewritten query system prompt.")

        rewritten_query_data = {
            "top_context": top_context,
            "original_query": original_query,
            "activities": activities,
            "kg_text": kg_text,
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

def llm_response_for_queryParts(llm, query_part, docs):
    try:
        activities = get_infoverve_activities()
        # logging.info("Activities related to automation:", activities)
        queryParts_data = {
            "activities": activities,
        }
        with open("data/prompts/main_system_prompt.txt", "r") as file:
            queryParts_system_prompt = file.read()
        

        queryParts_system_prompt = queryParts_system_prompt.format_map(queryParts_data)
        # Prepare the top context from the documents
        context = llmcontextBuilder(docs, collection_name, client)
        # "\n\n".join(doc[0].page_content for doc in docs)

        logging.info("Loaded rewritten query user prompt.")

        messages = [SystemMessage(content=queryParts_system_prompt), 
                    HumanMessage(content=f"Context:\n{context}\n\n need response for this query parts: {query_part}")]
        response = llm(messages)
        queryPart_response = response.content.strip()
    except Exception as e:
        logging.error(f"Error during query rewriting: {str(e)}")

    return queryPart_response


logging.info(".........................Starting Infoverve Helper Application.........................")
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
        if collection.name == "infoverve_docs_kg_hybrid":
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
    # logging.info(initial_docs_with_scores)
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
    llm_queryPart_responses = []
    for i, q in enumerate(query_parts):
        logging.info(f"Rewritten Query {i+1}: {q}")
        
        vec_rewritten = get_embedding(q)
        logging.info(f"Embedding generated for query {i+1}.")
        
        final_docs_with_score = vectorstore.similarity_search_with_score(query=q, k=no_of_response)

        
        logging.info(f"Processing query part {i+1}: {q}")
        queryPart_response = llm_response_for_queryParts(llm, q, final_docs_with_score)
        llm_queryPart_responses.append(queryPart_response)
        logging.info(f"Response for query part {i+1}: {queryPart_response}")

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
        context.extend([llmcontextBuilder(final_docs_with_score, collection_name, client)])


    logging.info(f"Found {len(context)} final documents.")
    logging.info("Context prepared for LLM response.")
    if len(llm_queryPart_responses) > 1:

        activities = get_infoverve_activities()
        # logging.info("Activities related to automation:", activities)
        main_data = {
            "activities": activities,
            "context": context,
            "query": query,
            "query_parts": query_parts,
            "llm_queryPart_responses": llm_queryPart_responses
        }
        with open("data/prompts/main_system_prompt.txt", "r") as file:
            main_system_prompt = file.read()
        

        main_system_prompt = main_system_prompt.format_map(main_data)
        
        with open("data/prompts/main_user_prompt.txt", "r") as file:
            main_user_prompt = file.read()
        

        main_user_prompt = main_user_prompt.format_map(main_data)
        messages = [
            SystemMessage(content=main_system_prompt),
            HumanMessage(content=main_user_prompt)
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
    else:
        response = llm_queryPart_responses[0]

        output_md_path = "./data/results/infoverve_helper_response.md"
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(response)
        logging.info("Single query part response:\n")
        logging.info(response)
        st.markdown(response)

