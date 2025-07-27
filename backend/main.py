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

# ------------------------- Setup -------------------------
data = json.load(open("data/products.json", encoding='utf-8'))
product_titles = [item['title'] for item in data]

nlp = spacy.load("en_core_web_sm")

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def translate_hindi_to_english(text: str) -> str:
    # TODO: Real translation here
    return normalize(text)

def extract_phrases(title: str):
    doc = nlp(title)
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = normalize(chunk.text.lower())
        if 1 <= len(chunk.text.split()) <= 4 and len(phrase) >= 2:
            phrases.append(phrase)
    return phrases

all_phrases = set()
for title in product_titles:
    all_phrases.update(extract_phrases(title))
suggestion_phrases = sorted(all_phrases)

phrase_phonetics = {phrase: doublemetaphone(phrase)[0] for phrase in suggestion_phrases}
product_titles_norm = [normalize(title) for title in product_titles]
product_titles_phonetic = [doublemetaphone(normalize(title))[0] for title in product_titles]

trending_keywords = ["rakhi", "smartphone", "headphones", "air conditioner", "washing machine"]

model = SentenceTransformer("all-MiniLM-L6-v2")
product_embeddings = model.encode(product_titles)
dimension = product_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(product_embeddings))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Product(BaseModel):
    id: int
    title: str
    price: float
    image: str
    rating: float


@app.get("/autosuggest", response_model=List[str])
def autosuggest(q: str = Query(..., min_length=1)):
    q_norm = normalize(q)
    q_translated = translate_hindi_to_english(q)
    if q_translated != q_norm:
        q_norm = q_translated

    q_phonetic = doublemetaphone(q_norm)[0]
    suggestions = []

    # 1. Trending keywords starting with q_norm
    trending_hits = [kw for kw in trending_keywords if kw.startswith(q_norm)]
    
    # For single letter, show all trending keywords starting with that letter with highest priority
    if len(q_norm) == 1:
        # Just ensure all trending keywords starting with this letter come first
        suggestions.extend(trending_hits)
    else:
        # For longer queries, trending keywords containing q_norm anywhere
        trending_hits_anywhere = [kw for kw in trending_keywords if q_norm in kw]
        suggestions.extend(trending_hits_anywhere)

    # 2. Startswith matches in phrases
    startswith_hits = [phrase for phrase in suggestion_phrases if phrase.startswith(q_norm) and phrase not in suggestions]

    # 3. Fuzzy matches
    if len(q_norm.split()) > 1:
        fuzzy_scorer = fuzz.token_set_ratio
        fuzzy_threshold = 50
    else:
        fuzzy_scorer = fuzz.partial_ratio
        fuzzy_threshold = 60

    fuzzy_matches = process.extract(q_norm, suggestion_phrases, scorer=fuzzy_scorer, limit=10)
    fuzzy_hits = [match[0] for match in fuzzy_matches if match[1] >= fuzzy_threshold and match[0] not in suggestions and match[0] not in startswith_hits]

    # 4. Substring matches
    substr_hits = [phrase for phrase in suggestion_phrases if q_norm in phrase and phrase not in suggestions and phrase not in startswith_hits]

    # 5. Phonetic matches
    phonetic_hits = [phrase for phrase, p_code in phrase_phonetics.items()
                     if p_code == q_phonetic and phrase not in suggestions and phrase not in startswith_hits]

    # Combine all:
    suggestions.extend(startswith_hits)
    suggestions.extend(fuzzy_hits)
    suggestions.extend(substr_hits)
    suggestions.extend(phonetic_hits)

    # Deduplicate and filter out 1-letter nonsense suggestions
    seen = set()
    result = []
    for s in suggestions:
        if s not in seen and len(s) > 1:
            seen.add(s)
            result.append(s)

    return result[:5]


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
    results.sort(key=lambda x: x[0], reverse=True)  # <-- fix here
    return [item for _, item in results]

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
