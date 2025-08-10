# migrate_qdrant_to_neo4j.py
import os
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from tqdm import tqdm

QDRANT_HOST = "ai.infoveave.cloud"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "infoverve_docs_kg_hybrid"
NEO4J_URI = "neo4j+s://034f8de5.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def upsert_triplet(tx, head, rel, tail, meta=None):
    # meta can contain {source_url, chunk_id...}
    tx.run(
        """
        MERGE (h:Entity {name: $head})
        MERGE (t:Entity {name: $tail})
        MERGE (h)-[r:RELATION {type: $rel}]->(t)
        SET r.count = coalesce(r.count, 0) + 1
        """,
        head=head, rel=rel, tail=tail
    )
    # Optionally attach source nodes/document linking if you want provenance
def create_entities(tx, entities):
    for ent in entities:
        tx.run(
            """
            MERGE (e:`{type}` {{name: $name}})
            """.format(type=ent["type"]),  # label is entity type
            name=ent["name"]
        )
def create_relationships(tx, triplets):
    for head, rel, tail in triplets:
        tx.run(
            """
            MATCH (a {{name: $head}}), (b {{name: $tail}})
            MERGE (a)-[r:`{rel}`]->(b)
            """.format(rel=rel),
            head=head,
            tail=tail
        )

def migrate():
    offset = None
    limit = 3000
    while True:
        points, offset = qclient.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=True,
            with_vectors=False,
            offset=offset,
            limit=limit
        )
        if not points:
            break

        with driver.session() as session:
            for p in tqdm(points):
                payload = p.payload or {}
                triplets = payload.get("triplets") or []   # expect list of (h, r, t)
                entities = payload.get("entities") or []   # optional
                url = payload.get("url")
                # Upsert each triplet
                session.write_transaction(create_entities, entities)
                session.write_transaction(create_relationships, triplets)

        if offset is None:
            break

if __name__ == "__main__":
    migrate()
    driver.close()
