from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re

options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

# Glossary Extraction from Infoveave Terminologies Page

glossary_url = "https://infoveave-help.pages.dev/introduction-to-infoveave/infoveave-terminologies/"

driver.get(glossary_url)
# time.sleep(2)  # Wait for JavaScript
soup = BeautifulSoup(driver.page_source, 'html.parser')

glossaries = []

for p_tag in soup.find_all('p'):
    Strong_tag = p_tag.find('strong')
    if Strong_tag:
        text = Strong_tag.get_text(strip=True)
        if text:
            glossaries.append(text)

print("Glossaries found: ",glossaries)

# Top-Level Link Extraction and Cleaning from Infoveave Getting Started Page

start_url = "https://infoveave-help.pages.dev/introduction-to-infoveave/getting-started/"       
driver.get(start_url)
parsed_soup = BeautifulSoup(driver.page_source, 'html.parser')
all_links = []
for anchor in parsed_soup.find_all('a', href=True):
    link_href = anchor['href']
    all_links.append(link_href)
top_level_pattern = r"^(/[^/]+/)"

top_level_paths = set(re.match(top_level_pattern, url).group(1) for url in all_links if re.match(top_level_pattern, url))
print("Top-level paths found: ", top_level_paths)
cleaned_paths = {
    re.sub(r'-v\d+$', '', path.strip('/'))  # strip slashes, remove -vN
    for path in top_level_paths
}

print(cleaned_paths)


# ----------------------------------------------------------
# Qdrant Collection Exploration and Semantic Search Example
# ----------------------------------------------------------

# from qdrant_client import QdrantClient
# # Connect to Qdrant — update host/port if you're not running locally
# client = QdrantClient(host="ai.infoveave.cloud", port=6333)

# # List all collections
# collections = client.get_collections()

# # Print collection names
# for collection in collections.collections:
#     collection_name = collection.name
#     print(f"Collection Name: {collection_name}")


# info = client.get_collection(collection_name=collection_name)
# print(info.dict())

# # Scroll through the collection
# response = client.scroll(
#     collection_name=collection_name,
#     with_payload=True,
#     with_vectors=True,   # Set to True if you want the actual embedding vectors too
#     limit=1             # Number of documents to fetch per request
# )
# print(f"Total points in collection '{collection_name}': {response[1]}")  # response[1] is the total number of points
# # Access the points (documents)
# points = response[0]  # response is a tuple: (List[records], next_offset)
# print(f"Number of points fetched: {points}")
# # for point in points:
# #     print("ID:", point.id)
# #     print("Payload:", point.payload)
# #     print("Vector:", point.vector)  # Only if with_vectors=True
# #     print("-----")
# from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# points = client.search(
#     collection_name="HelpDocuments",
#     query_vector=embedding_model.embed_query("How to create a Data Source?"),
#     limit=1,
#     with_payload=True
# )

# print(points[0].payload)

# --------------------------------------------------
# Streaming Chat Completion Example Using Groq API
# --------------------------------------------------
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