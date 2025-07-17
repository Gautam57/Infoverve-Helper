from groq import Groq
from qdrant_client import QdrantClient

# Connect to Qdrant — update host/port if you're not running locally
client = QdrantClient(host="ai.infoveave.cloud", port=6333)

# List all collections
collections = client.get_collections()

# Print collection names
for collection in collections.collections:
    collection_name = collection.name
    print(f"Collection Name: {collection_name}")


info = client.get_collection(collection_name=collection_name)
print(info.dict())

# Scroll through the collection
response = client.scroll(
    collection_name=collection_name,
    with_payload=True,
    with_vectors=True,   # Set to True if you want the actual embedding vectors too
    limit=1             # Number of documents to fetch per request
)
print(f"Total points in collection '{collection_name}': {response[1]}")  # response[1] is the total number of points
# Access the points (documents)
points = response[0]  # response is a tuple: (List[records], next_offset)
print(f"Number of points fetched: {len(points)}")
for point in points:
    print("ID:", point.id)
    print("Payload:", point.payload)
    print("Vector:", point.vector)  # Only if with_vectors=True
    print("-----")

# client = Groq()
# completion = client.chat.completions.create(
#     model="gemma2-9b-it",
#     messages=[
#       {
#         "role": "user",
#         "content": ""
#       }
#     ],
#     temperature=1,
#     max_completion_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")