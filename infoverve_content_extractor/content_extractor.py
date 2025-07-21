from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin, urldefrag, urlparse
from bs4 import BeautifulSoup
import time
import json
import re
from tqdm import tqdm

# Setup headless browser
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

Required_Info = []

glossary_url = "https://infoveave-help.pages.dev/introduction-to-infoveave/infoveave-terminologies/"
driver.get(glossary_url)
soup = BeautifulSoup(driver.page_source, 'html.parser')

glossaries = []

for p_tag in soup.find_all('p'):
    Strong_tag = p_tag.find('strong')
    if Strong_tag:
        term = Strong_tag.get_text(strip=True)
        if term:
            glossaries.append(term)

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

Required_Info.append({
    "terminologies": glossaries,
    "sections": list(cleaned_paths)
})

base_url = "https://infoveave-help.pages.dev/introduction-to-infoveave/getting-started/"

# Sets for visited and to-visit URLs
visited_urls = set()
to_visit = set([base_url])
# List to hold all results
result_data = []
counter = 0

pbar = tqdm(desc="Scraping pages")

while sorted(to_visit):
    current_url = to_visit.pop()
    if current_url in visited_urls:
        continue

    try:
        driver.get(current_url)
        time.sleep(2)  # Wait for JavaScript
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        visited_urls.add(current_url)

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']

            # Ignore links with #
            if '#' in href:
                continue

            full_url = urljoin(current_url, href)

            if full_url.startswith("https://infoveave-help.pages.dev") and full_url not in visited_urls:
                to_visit.add(full_url)

        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        for div in soup.find_all(attrs={'class': ['sidebar', 'toc', 'table-of-contents']}):
            div.decompose()

        text = soup.get_text(separator='\n', strip=True)
        parsed = urlparse(current_url)
        path = parsed.path.strip("/")  # "automation-v8/activities/clean-cache"
        if path.split("/")[-1] == "":
            title = "MainPage"
        else:
            title = path.split("/")[-1]
        
        terminologies = []

        for glossary in glossaries:
            if re.search(r'\b' + re.escape(glossary) + r'\b', text, re.IGNORECASE):
                terminologies.append(glossary)

        
        section = ''

        for i in top_level_paths:
            if i.lower() in current_url.lower():
                sections = re.sub(r'-v\d+$', '', i.strip('/'))

        word_count = len(text.split())

        
        result_data.append({
            "Sl_no":counter,
            "url": current_url,
            "Page_title":title,
            "section": sections,
            "no_of_char":len(text),
            "no_of_words": word_count,
            "Terminologies": terminologies,
            "content": text
        })
        counter +=1
        pbar.update(1)
    except Exception as e:
        print(f"Error accessing {current_url}: {e}")
        pbar.update(1)

driver.quit()

# Save to JSON file
with open("infoverve_content_extractor/data/infoveave_help_data.json", "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

with open("infoverve_content_extractor/data/infoveave_sections_and_terms.json", "w", encoding="utf-8") as f:
    json.dump(Required_Info, f, ensure_ascii=False, indent=2)

# Convert to list
unique_url_list = list(visited_urls)

# Output the list
print(f"\nTotal unique URLs found (excluding #): {len(unique_url_list)}\n")
for url in sorted(unique_url_list):
    print(url)
