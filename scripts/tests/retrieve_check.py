from qdrant_client import QdrantClient

client = QdrantClient(host="ai.infoveave.cloud", port=6333)

result = client.retrieve(
                collection_name="infoverve_helper_docs_hybrid",
                ids=["b4c2428f-df5e-454e-a95f-1b1b28e5afd1"],
                with_payload=True,
            )

print(result)