from fastapi import FastAPI, Query
from typing import List
from pydantic import BaseModel
import uvicorn
import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import process, fuzz
import spacy
from metaphone import doublemetaphone
# from argostranslate import translate  # Commented out due to installation issues

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Normalize text
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

# Translate Hindi to English (simplified version)
def translate_hindi_to_english(text: str) -> str:
    # For now, just return normalized text
    # TODO: Implement proper translation when argostranslate is available
    return normalize(text)

# Trending keywords
trending_keywords = ["rakhi", "smartphone", "headphones", "air conditioner", "washing machine"]

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load product catalog
data = json.load(open("data/products.json", encoding='utf-8'))
product_titles = [item['title'] for item in data]
product_titles_norm = [normalize(title) for title in product_titles]
product_titles_phonetic = [doublemetaphone(normalize(title))[0] for title in product_titles]

# Load sentence transformer and FAISS index  , do this :-> "apple earbuds" ≈ "AirPods" , "foldable phone" ≈ "Samsung Galaxy Z Fold"
model = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = model.encode(product_titles)
dimension = product_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(product_embeddings))

# Product schema
class Product(BaseModel):
    id: int
    title: str
    price: float
    image: str
    rating: float

# ------------------------- AUTOSUGGEST -------------------------
@app.get("/autosuggest", response_model=List[str])
def autosuggest(q: str = Query(..., min_length=1)):
    q_norm = normalize(q)
    q_translated = translate_hindi_to_english(q)
    if q_translated != q_norm:
        q_norm = q_translated

    q_phonetic = doublemetaphone(q_norm)[0]
    suggestions = []

    # 1. Titles that start with the query
    startswith_hits = [title for title in product_titles if normalize(title).startswith(q_norm)]

    # 2. Fuzzy matches (token-based, lower threshold for multi-word queries)
    if len(q_norm.split()) > 1:
        fuzzy_scorer = fuzz.token_set_ratio
        fuzzy_threshold = 50  # Lower threshold for multi-word queries
    else:
        fuzzy_scorer = fuzz.partial_ratio
        fuzzy_threshold = 60

    fuzzy_matches = process.extract(q_norm, product_titles, scorer=fuzzy_scorer, limit=10)
    fuzzy_hits = [match[0] for match in fuzzy_matches if match[1] >= fuzzy_threshold and match[0] not in startswith_hits]

    # 3. Substring matches (excluding already found)
    substr_hits = [title for title in product_titles if q_norm in normalize(title) and title not in startswith_hits]

    # 4. Phonetic matches (excluding already found)
    phonetic_hits = [title for i, title in enumerate(product_titles)
                     if product_titles_phonetic[i] == q_phonetic and title not in startswith_hits]

    # Combine, prioritizing startswith, then fuzzy
    suggestions.extend(startswith_hits)
    suggestions.extend(fuzzy_hits)
    suggestions.extend(substr_hits)
    suggestions.extend(phonetic_hits)

    # Remove duplicates, keep order
    seen, result = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            result.append(s)

    return result[:5]

# ------------------------- RANKING -------------------------
def rank_products(query_vec, candidates):
    results = []
    for item in candidates:
        title_vec = model.encode([item["title"]])[0]
        sim = np.dot(query_vec, title_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(title_vec))
        score = (
            0.5 * sim +
            0.3 * (item["rating"] / 5) +
            0.2 * (1 if normalize(item["title"]) in trending_keywords else 0)
        )
        results.append((score, item))
    results.sort(reverse=True)
    return [item for _, item in results]

# ------------------------- SEARCH -------------------------
@app.get("/search", response_model=List[Product])
def search(
    q: str = Query(..., min_length=1),
    min_price: float = 0,
    max_price: float = 1e6,
    min_rating: float = 0,
    brand: str = "",
    category: str = ""
):
    original_q = q
    q_norm = normalize(q)
    q_translated = translate_hindi_to_english(q)
    if q_translated != q_norm:
        q_norm = q_translated

    q_phonetic = doublemetaphone(q_norm)[0]

    doc = nlp(original_q)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            val = re.sub(r"[^\d]", "", ent.text)
            if val.isdigit():
                max_price = float(val)
        elif ent.label_ == "ORG":
            brand = ent.text.lower()
        elif ent.label_ in ["PRODUCT", "NORP"]:
            category = ent.text.lower()

    fuzzy_best = process.extractOne(q_norm, product_titles_norm, scorer=fuzz.partial_ratio)
    if fuzzy_best and fuzzy_best[1] >= 70:
        corrected_q = product_titles[product_titles_norm.index(fuzzy_best[0])]
        q_norm = normalize(corrected_q)

    if not fuzzy_best or fuzzy_best[1] < 60:
        for i, p in enumerate(product_titles_phonetic):
            if p == q_phonetic:
                q_norm = normalize(product_titles[i])
                break

    q_vec = model.encode([q_norm])
    D, I = index.search(np.array(q_vec), 50)
    candidates = [data[i] for i in I[0]]

    filtered = []
    for item in candidates:
        if not (min_price <= item['price'] <= max_price):
            continue
        if item['rating'] < min_rating:
            continue
        if brand and brand not in normalize(item['title']):
            continue
        if category and category not in normalize(item['title']):
            continue
        filtered.append(item)

    ranked = rank_products(q_vec[0], filtered)
    return ranked[:10]

# ------------------------- RUN -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
