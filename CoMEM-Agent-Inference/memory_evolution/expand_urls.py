import json
import random
import requests
from collections import defaultdict

# ------------------------
# Config
# ------------------------

INPUT_PATH = "GUI-Agent/categories_dict.json"
OUTPUT_PATH = "GUI-Agent/categories_dict_expand.json"
MAX_QUERIES_PER_SITE = 3
SEARCH_RESULTS_PER_QUERY = 10
MAX_SERPAPI_QUERIES = 200  # Budget limit

SERPAPI_KEY = ""

EXCLUDE_KEYWORDS = [
    "other site", "only use", "don't go", "account",
    "log in", "subscription", "must use"
]

# ------------------------
# SerpAPI DuckDuckGo Search Function
# ------------------------

def get_duckduckgo_links_serpapi(query, k=10):
    params = {
        "engine": "duckduckgo",
        "q": query,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = response.json()
        links = []
        for result in data.get("organic_results", []):
            if "link" in result:
                links.append(result["link"])
            if len(links) >= k:
                break
        print(f"Links for query: {query} | {links}")
        return links
    except Exception as e:
        print(f"SerpAPI error for query: {query} | {e}")
        return []

# ------------------------
# Query Filter
# ------------------------

def is_excludable(q):
    q = q.lower()
    return any(keyword in q for keyword in EXCLUDE_KEYWORDS)

# ------------------------
# Load and Sample Queries
# ------------------------

with open(INPUT_PATH) as f:
    data = json.load(f)

query_to_category = {}
sampled_queries = []

for category in data:
    for website_dict in data[category]:
        for website, queries in website_dict.items():
            filtered_queries = [q for q in queries if not is_excludable(q)]
            sampled = random.sample(filtered_queries, min(MAX_QUERIES_PER_SITE, len(filtered_queries)))
            for q in sampled:
                sampled_queries.append(q)
                query_to_category[q] = category

# Enforce SerpAPI budget
sampled_queries = sampled_queries[:MAX_SERPAPI_QUERIES]
print(f"Total sampled queries (within budget): {len(sampled_queries)}")

# ------------------------
# Sequential Link Fetching via SerpAPI
# ------------------------

query_links = {}

for query in sampled_queries:
    links = get_duckduckgo_links_serpapi(query, k=SEARCH_RESULTS_PER_QUERY)
    query_links[query] = links

# ------------------------
# Build Category → Links Dictionary
# ------------------------

expand_category_links = defaultdict(set)

for query, links in query_links.items():
    category = query_to_category[query]
    expand_category_links[category].update(links)

# Convert sets to lists for JSON
expand_category_links = {
    cat: list(links) for cat, links in expand_category_links.items()
}

# ------------------------
# Save Output
# ------------------------

with open(OUTPUT_PATH, "w") as f:
    json.dump(expand_category_links, f, indent=2)

print(f"Saved expanded category links to {OUTPUT_PATH}")