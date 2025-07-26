import json

# Load the JSON file
with open("data/infoverve_content_extractor/infoveave_help_data.json", "r") as f:
    data = json.load(f)

# Filter all entries with section == "automation"
automation_entries = [entry for entry in data if entry.get("section") == "automation"]

activities = []
# Print or use the content
for item in automation_entries:
    if "activities" in item['url'].split("/"):
        title = item['content'].split("|")[0]
        activities.append(title)

print("Activities related to automation:", activities)

        # print(f"URL: {item['url']}")
        # print(f"Title: {item['Page_title']}")
        # print(f"Content: {title}")
        # print("=" * 40)