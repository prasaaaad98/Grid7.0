from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
import json
import faiss
import numpy as np
import re
import math
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import process, fuzz
import spacy
from metaphone import doublemetaphone
import pygtrie
import networkx as nx
from collections import defaultdict, Counter
from functools import lru_cache

# ------------------------- Setup -------------------------
# NLP setup and normalization (define functions first)
nlp = spacy.load("en_core_web_sm")
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def stem_word(word: str) -> str:
    """Simple stemming to handle common plural forms"""
    if len(word) <= 3:
        return word
    
    # Common plural endings
    if word.endswith('s'):
        # Don't stem if it's already short or ends with 'ss'
        if len(word) > 3 and not word.endswith('ss'):
            return word[:-1]
    return word

def normalize_with_stemming(text: str) -> str:
    """Normalize text and apply stemming to each word"""
    normalized = normalize(text)
    words = normalized.split()
    stemmed_words = [stem_word(word) for word in words]
    return " ".join(stemmed_words)
def translate_hindi_to_english(text: str) -> str:
    return normalize(text)

# Load product data with extended schema
with open("data/products.json", encoding='utf-8') as f:
    data = json.load(f)

# Calculate max review count for normalization
MAX_REVIEWS = max(p.get('review_count', 0) for p in data) or 1

# ─── Build composite texts & embeddings ───
full_texts = []
for item in data:
    parts = [
        item["title"],
        item.get("description", ""),
        *item.get("synonyms", []),
        *item.get("tags", []),
        *item.get("subcategories", []),
        *item.get("category_path", [])
    ]
    full_texts.append(" ".join(parts))

model = SentenceTransformer("all-MiniLM-L6-v2")
title_embs = model.encode([item["title"] for item in data])
full_embs  = model.encode(full_texts)

dim = title_embs.shape[1]
title_index = faiss.IndexFlatL2(dim)
full_index  = faiss.IndexFlatL2(dim)
title_index.add(np.array(title_embs))
full_index.add(np.array(full_embs))

# ─── Build inverted index for lexical lookup ───
inv_index = defaultdict(set)
for idx, item in enumerate(data):
    text = normalize(item["title"]) + " " + normalize(item.get("description",""))
    for tok in text.split():
        inv_index[tok].add(idx)
        # Add stemmed version for plural/singular matching
        stemmed_tok = stem_word(tok)
        if stemmed_tok != tok:
            inv_index[stemmed_tok].add(idx)
    for syn in item.get("synonyms", []):
        for tok in normalize(syn).split():
            inv_index[tok].add(idx)
            stemmed_tok = stem_word(tok)
            if stemmed_tok != tok:
                inv_index[stemmed_tok].add(idx)
    for tag in item.get("tags", []):
        for tok in normalize(tag).split():
            inv_index[tok].add(idx)
            stemmed_tok = stem_word(tok)
            if stemmed_tok != tok:
                inv_index[stemmed_tok].add(idx)
    for subcat in item.get("subcategories", []):
        for tok in normalize(subcat).split():
            inv_index[tok].add(idx)
            stemmed_tok = stem_word(tok)
            if stemmed_tok != tok:
                inv_index[stemmed_tok].add(idx)
    for cat_path in item.get("category_path", []):
        for tok in normalize(cat_path).split():
            inv_index[tok].add(idx)
            stemmed_tok = stem_word(tok)
            if stemmed_tok != tok:
                inv_index[stemmed_tok].add(idx)

# ─── Build data map for related items ───
data_map = {item["id"]: item for item in data}

# Align fields and defaults
for item in data:
    if 'ratings' in item:
        item['rating'] = item.pop('ratings')
    item.setdefault('images', [])
    item.setdefault('retail_price', item.get('price', 0.0))
    item.setdefault('subcategories', [])
    item.setdefault('synonyms', [])
    item.setdefault('attributes', {})
    item.setdefault('tags', [])
    item.setdefault('related_ids', [])
    item.setdefault('contextual_data', {})
    item.setdefault('color', None)
    item.setdefault('material', None)
    item.setdefault('finish', None)
    item.setdefault('search_boost', 0.0)
    item.setdefault('assured_badge', False)

# Phrase extraction from title and description
def extract_phrases(text: str) -> List[str]:
    phrases = []
    doc = nlp(text or "")
    for chunk in doc.noun_chunks:
        tokens = chunk.text.lower().split()
        # skip chunks like "a X", "this Y", "the Z"
        if tokens[0] in {"a","an","the","this","that","these","those"}:
            continue
        phrase = normalize(chunk.text)
        if 1 <= len(chunk.text.split()) <= 4 and len(phrase) >= 2:
            phrases.append(phrase)
    return phrases

# Build suggestion phrases set
suggestion_phrases = set()
for item in data:
    title = item.get('title', '')
    desc = item.get('description', '')
    suggestion_phrases.update(extract_phrases(title))
    suggestion_phrases.update(extract_phrases(desc))
    for tok in normalize(title).split():
        if len(tok) > 1:
            suggestion_phrases.add(tok)
    for syn in item.get('synonyms', []):
        ph = normalize(syn)
        if ph: suggestion_phrases.add(ph)
    for tag in item.get('tags', []):
        ph = normalize(tag)
        if ph: suggestion_phrases.add(ph)
    for val in item.get('attributes', {}).values():
        ph = normalize(str(val))
        if ph: suggestion_phrases.add(ph)
    # Add contextual_data values
    for val in item.get('contextual_data', {}).values():
        ph = normalize(val)
        if ph:
            suggestion_phrases.add(ph)
    # Add color, material, and finish
    for extra in ('color', 'material', 'finish'):
        value = item.get(extra)
        if value:
            ph = normalize(str(value))
            suggestion_phrases.add(ph)
suggestion_phrases = sorted(suggestion_phrases)

# Prefix Trie
phrase_trie = pygtrie.CharTrie()
for ph in suggestion_phrases:
    phrase_trie[ph] = None

# Phonetic map
phrase_phonetics = {ph: doublemetaphone(ph)[0] for ph in suggestion_phrases}

# Semantic models and FAISS indexes
model = SentenceTransformer("all-MiniLM-L6-v2")
# Title embeddings for search
titles = [item['title'] for item in data]
title_embs = model.encode(titles)
dim = title_embs.shape[1]
title_index = faiss.IndexFlatL2(dim)
title_index.add(np.array(title_embs))
# Phrase embeddings for autosuggest
phrase_embs = model.encode(suggestion_phrases)
ph_dim = phrase_embs.shape[1]
phrase_index = faiss.IndexFlatL2(ph_dim)
phrase_index.add(np.array(phrase_embs))

# Build phrase->product mapping for fast KG edges
phrase_to_products = defaultdict(list)
for idx, item in enumerate(data):
    text = " ".join([
        normalize(item.get('title','')),
        normalize(item.get('description','')),
        *[normalize(s) for s in item.get('synonyms', [])],
        *item.get('tags', []),
        *[str(v) for v in item.get('attributes', {}).values()],
        *[str(v) for v in item.get('contextual_data', {}).values()],
        str(item.get('color','')),
        str(item.get('finish',''))
    ])
    for ph in suggestion_phrases:
        if ph in text:
            phrase_to_products[ph].append(idx)

# Build simple knowledge graph
G = nx.Graph()
# Add phrase nodes
for ph in suggestion_phrases:
    G.add_node(ph, type='phrase')
# Add category, synonyms, tags, attributes nodes and edges
for ph, idxs in phrase_to_products.items():
    for idx in idxs:
        item = data[idx]
        # category_path edges
        for parent, child in zip(item.get('category_path', []), item.get('category_path', [])[1:]):
            G.add_node(parent, type='category')
            G.add_node(child, type='category')
            G.add_edge(parent, child)
            G.add_edge(ph, parent)
        # synonyms
        for syn in item.get('synonyms', []):
            psyn = normalize(syn)
            G.add_node(psyn, type='synonym')
            G.add_edge(ph, psyn)
        # tags
        for tag in item.get('tags', []):
            t = normalize(tag)
            G.add_node(t, type='tag')
            G.add_edge(ph, t)
        # attributes
        for val in item.get('attributes', {}).values():
            av = normalize(str(val))
            G.add_node(av, type='attribute')
            G.add_edge(ph, av)
        # contextual_data nodes
        for val in item.get('contextual_data', {}).values():
            node = normalize(val)
            G.add_node(node, type='contextual')
            G.add_edge(ph, node)
        # color & finish nodes
        for field in ('color', 'finish'):
            val = item.get(field)
            if val:
                node = normalize(str(val))
                G.add_node(node, type=field)
                G.add_edge(ph, node)

# When user hasn't typed anything yet, show these top‐level trends:
global_trending = [
    "mobiles", "shoes", "t shirts", "laptops",
    "watches", "tv", "sarees", "earbuds"
]

# Hardcoded trending per letter (define for A-Z)
import string
trending_map = {c: [] for c in string.ascii_lowercase}
trending_map = {
    'a': [
        "Ai nova 5g",
        "Anarkali suit women",
        "Asus vivobook 15",
        "Airpods",
        "Ac 1 ton",
        "Almirah",
        "Abros shoes",
        "Axor helmets",
    ],
    'b': [
        "Boat earbuds",
        "Bag for men",
        "Bluetooth",
        "Bluetooth headphone",
        "Birthday decorations kit",
        "Bare anatomy shampoo",
        "Buds",
        "Bldc ceiling fan",
    ],
    'c': [
        "Cycle",
        "Creatine",
        "Crocs men",
        "Cctv camera",
        "Cord set women",
        "Creatine monohydrate",
        "Clogs for men",
        "Counter name jaap",
    ],
    'd': [
        "Dermdoc night cream",
        "Drone camera",
        "Dinner set",
        "Deconstruct sunscreen",
        "Dresses for women",
        "Drill machine",
        "Dell laptop",
        "Dry fruits combo",
    ],
    'e': [
        "Earbuds",
        "Extension board",
        "Earrings for women",
        "Ethiglo face wash",
        "Earphone",
        "Electric kettle",
        "Egg boilers",
        "Edge 60 pro",
    ],
    'f': [
        "Flats for women",
        "Fridge single door",
        "Fishing rod",
        "Face wash women",
        "Face wash",
        "Footwear for women",
        "Fastrack watches men",
        "Fire boltt smartwatch",
    ],
    'g': [
        "Ghar soap",
        "Gharsoap magic soap",
        "Guitar",
        "G85 moto",
        "Gas stove",
        "Google pixel 9a",
        "Goggles men",
        "Gym set",
    ],
    'h': [
        "Headphone",
        "Hp laptop",
        "Hand bags women",
        "Heels for women",
        "Himalaya face wash",
        "Helmet",
        "Hookah",
        "Home theatre speakers",
    ],
    'i': [
        "Iphone 16",
        "Iphone 15",
        "Iqoo z10 5g",
        "Iphone 13",
        "Infinix mobile 5g",
        "Infinix 50s 5g",
        "Iqoo neo 10",
        "Iqoo mobile",
    ],
    'j': [
        "Jeans men",
        "Jumpsuit for women",
        "Jio phone",
        "Jeans for women",
        "Juicer mixer grinder",
        "Jacket for men",
        "Jockey underwear",
        "Jewellery set",
    ],
    'k': [
        "Kurti pant set",
        "Kurti for women",
        "Kurta set women",
        "Krishna dress set",
        "Kozicare soap",
        "Keypad mobiles",
        "Kurti",
        "K13x 5g",
    ],
    'l': [
        "Laptop",
        "Lipstick",
        "Loreal shampoo",
        "Loafers for men",
        "Lunch box",
        "Libras kurta set",
        "Lehenga for women",
        "Lava 5g mobile",
    ],
    'm': [
        "Mobile 5g",
        "Motorola 5g mobile",
        "Motorola 60 fusion",
        "Muuchstac face wash",
        "Mosquito killer bat",
        "Moto g96 5g",
        "Moto g85 5g",
        "Motorola g96 5g",
    ],
    'n': [
        "Nothing phone 3a",
        "Nothing mobile",
        "Necklace for women",
        "Night suit",
        "Noise earbuds",
        "Noise smartwatch",
        "Nighty for women",
        "Nike shoes men",
    ],
    'o': [
        "Oppo reno14 pro",
        "Oneplus mobile 5g",
        "Oppe k13 5g",
        "One plus mobile",
        "Oneplus earbuds",
        "Office chair",
        "Oats",
    ],
    'p': [
        "Poco 5g mobile",
        "Pens",
        "Pants for men",
        "Pilgrim face serum",
        "Poco f7 5g",
        "Photo frames",
        "Power bank",
        "Phone",
        "Projector",
    ],
    'q': [
        "Qioo new mobile",
        "Queen size bed",
        "Quaker oats",
        "Quinoa",
        "Iphone 13",
        "Queen size mattress",
        "Queen size bedsheets",
        "Quechua bags",
    ],
    'r': [
        "Rakhi set",
        "Realme 5g mobile",
        "Redmi 5g mobile",
        "Rakhi for brothers",
        "Rakhi",
        "Raincoat",
        "Remote control car",
        "Realme p3 5g",
    ],
    's': [
        "Samsung 5g mobiles",
        "Shirt for men",
        "Smart watches",
        "Samsung s24 ultra",
        "S24 ultra mobile",
        "Saree new design",
        "Shoes for men",
        "Saree",
    ],
    't': [
        "T shirt",
        "Tote bags women",
        "Tops for women",
        "Trimmer men",
        "Tablet 5g",
        "Tiffin box",
        "Teddy bear",
        "Trimmer men phillips",
    ],
    'u': [
        "Umbrella",
        "Umbrella",
        "Underwear men",
        "Ultra 24 samsung",
        "Urbanguru hair removal",
        "Uno cards",
        "S24 ultra mobile",
        "Ups",
    ],
    'v': [
        "Vivo 5g mobiles",
        "Vivo x200 fe",
        "Vivo v50",
        "Vitamin c serum",
        "Vivo t4x",
        "Vivo y29 5g",
        "Vivo t4 5g",
        "Vivo x200 pro",
    ],
    'w': [
        "Watch men",
        "Watch for women",
        "Water bottle",
        "Wishcare hair serum",
        "Wild stone perfume",
        "Washing machine automatic",
        "Wardrobe",
        "Wallpaper for wall",
    ],
    'x': [
        "Xyxx underwear men",
        "X200 pro",
        "Vivo x200 fe",
        "Xtreme 125r",
        "X200 fe",
        "Xiaomi 14 ultra",
        "X200",
        "Xr",
    ],
    'y': [
        "Yoga mat",
        "Y29 5g",
        "Y400 pro vivo",
        "Y19 5g",
        "Y39 vivo",
        "Yoga mat women",
        "Yellow saree",
        "Y400 vivo mobile",
    ],
    'z': [
        "Zebronics soundbars",
        "Z9s mobile",
        "Zouk bags women",
        "Z fold 7",
        "Z10x 5g",
        "Zodiac imp backpack",
        "Zee raincoat waterproof",
        "Zebronics headphones",
    ],
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class Product(BaseModel):
    id: int
    title: str
    brand: str
    category: str
    price: float
    retail_price: Optional[float]
    images: List[str]
    rating: float
    description: str
    category_path: List[str]
    subcategories: List[str]
    synonyms: List[str]
    tags: List[str]
    attributes: dict
    contextual_data: Optional[dict] = {}
    color: Optional[str] = None
    material: Optional[str] = None
    finish: Optional[str] = None
    search_boost: Optional[float] = 0.0
    related_ids: List[int]
    assured_badge: Optional[bool] = False
    class Config:
        allow_population_by_field_name = True

@app.get("/autosuggest", response_model=List[str])
def autosuggest(
    q: str = Query("", min_length=0),
    context_category_path: Optional[str] = Query(None, description="Previous category path as JSON string for context-aware suggestions")
):
    qn = normalize(q)
    qt = translate_hindi_to_english(qn)
    if qt != qn:
        qn = qt

    # Parse context category path
    previous_category_path = []
    if context_category_path:
        try:
            import json
            previous_category_path = json.loads(context_category_path)
        except:
            previous_category_path = []

    # 0-char: show global trending
    if qn == "":
        return global_trending

    # 1-char: per-letter trending
    if len(qn) == 1:
        return trending_map.get(qn, [])[:8]
    
    suggestions = []
    context_suggestions = []
    # trending for prefix
    suggestions += [t for t in trending_map.get(qn[0], []) if normalize(t).startswith(qn)]
    # trie
    try:
        suggestions += [ph for ph in phrase_trie.keys(prefix=qn) if ph not in suggestions]
    except KeyError:
        pass
    # substring (whole-word match)
    substr_hits = []
    for ph in suggestion_phrases:
        if re.search(rf"\b{re.escape(qn)}\b", ph) and ph not in suggestions:
            substr_hits.append(ph)
        # Also check stemmed version
        stemmed_qn = " ".join([stem_word(word) for word in qn.split()])
        if stemmed_qn != qn and re.search(rf"\b{re.escape(stemmed_qn)}\b", ph) and ph not in suggestions:
            substr_hits.append(ph)
    suggestions += substr_hits
    # fuzzy (only if trie and substring didn't find enough)
    fuzzy_hits = []
    if len(suggestions) < 3:  # Only run fuzzy if we need more suggestions
        if len(qn.split()) == 1:
            fuzzy_scorer = fuzz.partial_ratio  # more forgiving for single tokens
            fuzzy_threshold = 60
        else:
            fuzzy_scorer = fuzz.token_set_ratio
            fuzzy_threshold = 60
            
        fam = process.extract(qn, suggestion_phrases, scorer=fuzzy_scorer, limit=10)
        fuzzy_hits = [m[0] for m in fam if m[1] >= fuzzy_threshold and m[0] not in suggestions]
        
        # only keep phonetically similar terms
        q_phonetic = doublemetaphone(qn)[0]
        fuzzy_hits = [
            ph for ph in fuzzy_hits
            if phrase_phonetics.get(ph) == q_phonetic
        ]
        
        suggestions += fuzzy_hits
    # phonetic
    code = doublemetaphone(qn)[0]
    suggestions += [ph for ph,c in phrase_phonetics.items() if c == code and ph not in suggestions]
    # semantic phrase neighbors—but only true extensions when user typed >1 word
    q_emb = model.encode([qn])
    _, idx = phrase_index.search(np.array(q_emb), 10)
    sem_hits = []
    prefix_tokens = qn.split()
    full_prefix = " ".join(prefix_tokens)
    for i in idx[0]:
        ph = suggestion_phrases[i]
        # skip if already in list
        if ph in suggestions:
            continue
        # if multi-word prefix, demand exact phrase-prefix match
        if len(prefix_tokens) > 1:
            if not ph.startswith(full_prefix):
                continue
        sem_hits.append(ph)
        if len(sem_hits) >= 5:
            break
    suggestions += sem_hits
    # graph expansions (expand only from top candidates, and only on‑prefix neighbors)
    seed = suggestions[:5].copy()
    added = 0
    for seed_ph in seed:
        if not G.has_node(seed_ph):
            continue
        for neigh in G.neighbors(seed_ph):
            # only add if it starts with the same prefix and isn’t already in suggestions
            if neigh.startswith(qn) and neigh not in suggestions:
                suggestions.append(neigh)
                added += 1
                if added >= 2:
                    break
        if added >= 2:
            break

    # ===== KG expansions & templating (seed-based) =====
    # Extract the "seed" word (first token) from the normalized prefix
    tokens = qn.split()
    seed = tokens[0] if tokens else qn

    templated = []
    if G.has_node(seed):
        for neigh in G.neighbors(seed):
            ntype = G.nodes[neigh]['type']
            # Build the appropriate template
            if ntype in ('tag', 'contextual'):
                cand = f"{seed} for {neigh}"
            elif ntype in ('attribute', 'material', 'finish'):
                cand = f"{seed} with {neigh}"
            elif ntype == 'category':
                cand = f"{seed} in {neigh}"
            else:
                continue

            # Only include templates that match what the user has typed so far
            if cand.startswith(qn) and cand not in templated:
                templated.append(cand)

    # Inject up to two templates
    suggestions.extend(templated[:2])

    # Mark all template suggestions for bonus scoring
    template_set = set(templated[:2])

    # Enhanced price-range templates and detection
    tokens = qn.split()
    
    # Detect price-related queries
    price_keywords = ["under", "below", "less than", "cheap", "budget", "affordable"]
    price_patterns = [
        r"under\s+(\d+)",
        r"below\s+(\d+)", 
        r"less\s+than\s+(\d+)",
        r"cheap\s+(\w+)",
        r"budget\s+(\w+)",
        r"affordable\s+(\w+)"
    ]
    
    # Check if query contains price-related keywords
    has_price_keyword = any(keyword in qn for keyword in price_keywords)
    
    # Extract base product category from query
    product_categories = {
        "mobile": ["phone", "mobile", "smartphone", "iphone", "samsung", "oneplus", "vivo", "motorola"],
        "laptop": ["laptop", "computer", "dell", "hp", "lenovo", "macbook"],
        "shoes": ["shoes", "footwear", "sneakers", "boots"],
        "tshirts": ["tshirt", "t-shirt", "shirt", "clothing"],
        "headphones": ["headphones", "earbuds", "airpods", "audio"],
        "watch": ["watch", "smartwatch", "apple watch"],
        "accessories": ["cover", "case", "mobile cover", "phone case"]
    }
    
    detected_category = None
    for category, keywords in product_categories.items():
        if any(keyword in qn for keyword in keywords):
            detected_category = category
            break
    
    # Define price ranges based on category
    category_price_ranges = {
        "mobile": [5000, 10000, 15000, 20000, 30000, 50000],
        "laptop": [20000, 30000, 40000, 50000, 70000, 100000],
        "shoes": [500, 1000, 2000, 3000, 5000],
        "tshirts": [500, 1000, 1500, 2000, 3000],
        "headphones": [1000, 2000, 5000, 10000, 20000],
        "watch": [5000, 10000, 20000, 30000, 50000],
        "accessories": [500, 1000, 2000, 3000],
        "default": [500, 1000, 2000, 5000, 10000]
    }
    
    # If user types "under" as last word, suggest price ranges
    if tokens and tokens[-1] == "under":
        base = " ".join(tokens[:-1]).strip()
        price_ranges = category_price_ranges.get(detected_category, category_price_ranges["default"])
        
        for bucket in price_ranges:
            cand = f"{base} under {bucket}"
            if cand not in suggestions:
                suggestions.append(cand)
    
    # If user types partial price after "under", suggest complete prices
    if tokens and len(tokens) >= 2 and tokens[-2] == "under":
        partial_price = tokens[-1]
        base = " ".join(tokens[:-2]).strip()
        
        # Check if partial_price is a number
        if partial_price.isdigit():
            price_ranges = category_price_ranges.get(detected_category, category_price_ranges["default"])
            
            # Find price ranges that start with the partial price
            matching_prices = []
            for bucket in price_ranges:
                if str(bucket).startswith(partial_price):
                    matching_prices.append(bucket)
            
            # If no exact matches, suggest some reasonable prices
            if not matching_prices:
                partial_int = int(partial_price)
                # Suggest prices around the partial input
                if partial_int < 1000:
                    matching_prices = [partial_int * 1000, partial_int * 1000 + 5000, partial_int * 1000 + 10000]
                elif partial_int < 10000:
                    matching_prices = [partial_int * 1000, partial_int * 1000 + 5000, partial_int * 1000 + 10000]
                else:
                    matching_prices = [partial_int * 1000, partial_int * 1000 + 5000, partial_int * 1000 + 10000]
            
            for bucket in matching_prices[:3]:  # Limit to 3 suggestions
                cand = f"{base} under {bucket}"
                if cand not in suggestions:
                    suggestions.append(cand)
    
    # If query contains price-related keywords but no specific price, suggest price ranges
    elif has_price_keyword and detected_category:
        # Extract the product term (remove price keywords)
        product_term = qn
        for keyword in price_keywords:
            product_term = product_term.replace(keyword, "").strip()
        
        if product_term:
            price_ranges = category_price_ranges.get(detected_category, category_price_ranges["default"])
            for bucket in price_ranges:
                cand = f"{product_term} under {bucket}"
                if cand not in suggestions:
                    suggestions.append(cand)
    
    # Enhanced suggestions for product categories with price ranges
    elif detected_category and len(qn.split()) <= 2:  # Short queries like "mobile", "laptop"
        price_ranges = category_price_ranges.get(detected_category, category_price_ranges["default"])
        # Suggest top 3 most relevant price ranges for the category
        for bucket in price_ranges[:3]:
            cand = f"{qn} under {bucket}"
            if cand not in suggestions:
                suggestions.append(cand)
    
    # If query contains a specific price, suggest similar price ranges
    price_match = re.search(r'(\d+)', qn)
    if price_match and detected_category:
        price = int(price_match.group(1))
        price_ranges = category_price_ranges.get(detected_category, category_price_ranges["default"])
        
        # Find relevant price ranges around the mentioned price
        relevant_ranges = [r for r in price_ranges if r >= price * 0.5 and r <= price * 2]
        if not relevant_ranges:
            relevant_ranges = price_ranges[:3]  # Default to first 3 ranges
        
        for bucket in relevant_ranges:
            # Extract base product term
            base = re.sub(r'\d+', '', qn).strip()
            if base:
                cand = f"{base} under {bucket}"
                if cand not in suggestions:
                    suggestions.append(cand)

    # brand-prefix expansions (bring in `<brand> <prefix>`)
    brands_seen = set()
    for item in data:
        if qn in normalize(item['title']):
            bp = f"{item['brand'].lower()} {qn}"
            if bp not in suggestions and bp not in brands_seen:
                suggestions.append(bp)
                brands_seen.add(bp)
            if len(brands_seen) >= 3:
                break

    # Compute comprehensive quality scores for suggestions
    scores = {}
    for ph in suggestions:
        prod_ids = phrase_to_products.get(ph, [])
        if not prod_ids:
            scores[ph] = 0
            continue

        # take the max boost among them
        boost_val = max(data[i].get('search_boost', 0) for i in prod_ids)
        # best rating normalized to 0–1
        rating_val = max(data[i].get('rating', 0) for i in prod_ids) / 5.0
        # most-reviewed product, log-scaled 0–1
        reviews = max(data[i].get('review_count', 0) for i in prod_ids)
        review_val = math.log1p(reviews) / math.log1p(MAX_REVIEWS)

        # combine with w1+w2+w3 = 1 (boosted search_boost for brand-driven corrections)
        scores[ph] = 0.7 * boost_val + 0.2 * rating_val + 0.1 * review_val

        # *** TEMPLATE BOOST ***
        if ph in template_set:
            scores[ph] += 1.0   # give a flat +1 score boost

    # 1) Separate out the templates
    templates = [ph for ph in suggestions if ph in template_set]

    # 2) Collect all other suggestions
    others = [ph for ph in suggestions if ph not in template_set]

    # 3) Sort the "others" by quality score
    others.sort(key=lambda ph: scores.get(ph, 0), reverse=True)

    # 4) Build the final list: templates first, then top-ranked others
    max_suggestions = 8
    final = templates[:2] + others[: (max_suggestions - len(templates[:2]))]

    # ===== CONTEXT-AWARE PRIORITIZATION =====
    # If we have previous category path, prioritize suggestions from same hierarchy
    if previous_category_path and len(previous_category_path) >= 2:
        context_suggestions = []
        regular_suggestions = []
        
        # Check each suggestion against products to see if they share category context
        for suggestion in final:
            suggestion_norm = normalize(suggestion)
            has_context_match = False
            
            # Look for products that match this suggestion
            for item in data:
                item_title_norm = normalize(item.get('title', ''))
                item_desc_norm = normalize(item.get('description', ''))
                item_category_path = item.get('category_path', [])
                
                # Check if suggestion matches this product (title, description, or synonyms)
                suggestion_matches_product = (
                    suggestion_norm in item_title_norm or 
                    any(suggestion_norm in normalize(syn) for syn in item.get('synonyms', [])) or
                    suggestion_norm in item_desc_norm
                )
                
                if suggestion_matches_product:
                    # Check if this product shares category hierarchy with previous search
                    # Match first 2 levels: e.g., ["Apparel", "Men"] from ["Apparel", "Men", "Shirts", "Denim Shirts"]
                    if (len(item_category_path) >= 2 and 
                        len(previous_category_path) >= 2 and
                        item_category_path[0] == previous_category_path[0] and  # Same top category
                        item_category_path[1] == previous_category_path[1]):   # Same second level
                        has_context_match = True
                        break
            
            if has_context_match:
                context_suggestions.append(suggestion)
            else:
                regular_suggestions.append(suggestion)
        
        # Reorder: context-matched suggestions first, then regular ones
        final = context_suggestions + regular_suggestions

    # 5) Dedupe & Title-case
    seen_norm = set()
    res = []
    for s in final:
        sn = normalize(s)
        if sn not in seen_norm and len(s) > 1:
            seen_norm.add(sn)
            res.append(s.title())

    return res

# Remove caching for now to debug sorting
def _compute_search(q_norm, min_price, max_price, min_rating, brand, category, sort_by="relevance"):
    # 1. Encode normalized query
    q_emb = model.encode([q_norm])

    # 2. FAISS on title and full texts
    _, I1 = title_index.search(np.array(q_emb), 50)
    _, I2 = full_index.search(np.array(q_emb), 50)
    faiss_ids = set(I1[0]) | set(I2[0])

    # 3. Lexical hits via inverted index
    lex_ids = set()
    for tok in q_norm.split():
        lex_ids |= inv_index.get(tok, set())
        # Also search for stemmed version
        stemmed_tok = stem_word(tok)
        if stemmed_tok != tok:
            lex_ids |= inv_index.get(stemmed_tok, set())

    cand_ids = faiss_ids | lex_ids
    candidates = [data[i] for i in cand_ids]

    # Filters
    filtered = []
    for item in candidates:
        if not (min_price <= item["price"] <= max_price): continue
        if item["rating"] < min_rating: continue
        if brand and brand.lower() not in normalize(item["brand"]): continue
        if category and category.lower() not in normalize(item["category"]): continue
        filtered.append(item)

    # Precompute for scoring
    MAX_REVIEWS = max(it.get("review_count",0) for it in data) or 1

    # Score each
    scored = []
    for item in filtered:
        # semantic sim (title only)
        title_vec = model.encode([item["title"]])[0]
        sim = np.dot(q_emb, title_vec) / (np.linalg.norm(q_emb) * np.linalg.norm(title_vec))

        rating_norm = item["rating"] / 5.0
        boost_norm = item.get("search_boost", 0) / 100.0
        review_norm = math.log1p(item.get("review_count",0)) / math.log1p(MAX_REVIEWS)

        score = 0.4 * sim + 0.2 * rating_norm + 0.2 * boost_norm + 0.2 * review_norm
        scored.append((score, item))

    # Apply sorting based on sort_by parameter
    print(f"Sorting by: {sort_by}")
    print(f"Before sorting - first 3 products: {[(item[1]['title'], item[1]['price']) for item in scored[:3]]}")
    
    if sort_by == "price_low_high":
        scored.sort(key=lambda x: x[1]["price"])
    elif sort_by == "price_high_low":
        scored.sort(key=lambda x: x[1]["price"], reverse=True)
    elif sort_by == "rating":
        scored.sort(key=lambda x: x[1]["rating"], reverse=True)
    elif sort_by == "newest":
        # Assuming newer products have higher IDs (you might need to adjust this based on your data)
        scored.sort(key=lambda x: x[1]["id"], reverse=True)
    else:  # relevance (default)
        scored.sort(key=lambda x: x[0], reverse=True)  # Sort by relevance score
    
    print(f"After sorting - first 3 products: {[(item[1]['title'], item[1]['price']) for item in scored[:3]]}")

    # Related-IDs interleaving
    final_list = []
    for score, item in sorted(scored, key=lambda x: x[0], reverse=True):
        final_list.append(item)
        for rid in item.get("related_ids", [])[:2]:
            related = data_map.get(rid)
            if related and related not in final_list:
                final_list.append(related)
        if len(final_list) >= 10:
            break

    # Identify sponsored product (highest search_boost, or least related if tied)
    # Exclude the first product (most relevant) from sponsored consideration
    sponsored_id = None
    if len(final_list) > 1:  # Need at least 2 products to have a sponsored one
        # Consider all products except the first one (most relevant)
        remaining_products = final_list[1:]
        
        # Find products with highest search_boost among remaining products
        max_boost = max(item.get("search_boost", 0) for item in remaining_products)
        high_boost_items = [item for item in remaining_products if item.get("search_boost", 0) == max_boost]
        
        if len(high_boost_items) == 1:
            # Only one product with highest boost
            sponsored_id = high_boost_items[0]["id"]
        elif len(high_boost_items) > 1:
            # Multiple products with same boost, find least related to query
            q_emb = model.encode([q_norm])
            least_related = None
            lowest_sim = 1.0
            
            for item in high_boost_items:
                title_vec = model.encode([item["title"]])[0]
                sim = np.dot(q_emb, title_vec) / (np.linalg.norm(q_emb) * np.linalg.norm(title_vec))
                if sim < lowest_sim:
                    lowest_sim = sim
                    least_related = item
            
            sponsored_id = least_related["id"] if least_related else None

    # Facets & total-hits
    total_hits = len(filtered)
    brand_facet = Counter(it["brand"] for it in filtered)
    cat_facet   = Counter(it["category"] for it in filtered)

    def serialize(item):
        return {
            'id': item['id'],
            'title': item['title'],
            'brand': item['brand'],
            'category': item['category'],
            'price': item['price'],
            'retail_price': item['retail_price'],
            'images': item['images'],
            'rating': item['rating'],
            'description': item['description'],
            'category_path': item.get('category_path', []),
            'subcategories': item.get('subcategories', []),
            'synonyms': item.get('synonyms', []),
            'tags': item.get('tags', []),
            'attributes': item.get('attributes', {}),
            'contextual_data': item.get('contextual_data', {}),
            'color': item.get('color'),
            'material': item.get('material'),
            'finish': item.get('finish'),
            'search_boost': item.get('search_boost', 0.0),
            'related_ids': item.get('related_ids', []),
            'assured_badge': item.get('assured_badge', False),
            'warehouse_loc': item.get('warehouse_loc'),
            'is_high_rated_alternative': item.get('is_high_rated_alternative', False)
            # Only warehouse coordinates - delivery calculated dynamically on frontend
        }

    # Extract category_path from the most relevant product (first in results)
    most_relevant_category_path = final_list[0].get('category_path', []) if final_list else []
    
    return {
        "total_hits": total_hits,
        "sponsored_id": sponsored_id,
        "results": [serialize(p) for p in final_list[:10]],
        "most_relevant_category_path": most_relevant_category_path,
        "facets": {
            "brands": brand_facet.most_common(),
            "categories": cat_facet.most_common()
        }
    }

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    min_price: float = 0,
    max_price: float = 1e6,
    min_rating: float = 0,
    brand: str = "",
    category: str = "",
    sort: str = "relevance"
):
    # normalize & translate
    q_orig = q
    q_norm = normalize(q)
    qt = translate_hindi_to_english(q_norm)
    if qt != q_norm:
        q_norm = qt
    
    # Enhanced price parsing from query
    
    # Parse price-related queries
    price_keywords = ["under", "below", "less than", "cheap", "budget", "affordable"]
    price_patterns = [
        r"under\s+(\d+)",
        r"below\s+(\d+)", 
        r"less\s+than\s+(\d+)",
        r"cheap\s+(\w+)",
        r"budget\s+(\w+)",
        r"affordable\s+(\w+)"
    ]
    
    # Extract price from query if present
    extracted_price = None
    for pattern in price_patterns:
        match = re.search(pattern, q_norm)
        if match:
            if pattern in [r"under\s+(\d+)", r"below\s+(\d+)", r"less\s+than\s+(\d+)"]:
                extracted_price = int(match.group(1))
                break
    
    # Detect product category for price range adjustment
    product_categories = {
        "mobile": ["phone", "mobile", "smartphone", "iphone", "samsung", "oneplus", "vivo", "motorola"],
        "laptop": ["laptop", "computer", "dell", "hp", "lenovo", "macbook"],
        "shoes": ["shoes", "footwear", "sneakers", "boots"],
        "tshirts": ["tshirt", "t-shirt", "shirt", "clothing"],
        "headphones": ["headphones", "earbuds", "airpods", "audio"],
        "watch": ["watch", "smartwatch", "apple watch"],
        "accessories": ["cover", "case", "mobile cover", "phone case"]
    }
    
    detected_category = None
    for category_name, keywords in product_categories.items():
        if any(keyword in q_norm for keyword in keywords):
            detected_category = category_name
            break
    
    # Adjust price filtering based on extracted price and category
    adjusted_min_price = min_price
    adjusted_max_price = max_price
    
    if extracted_price:
        # For "under X" queries, set max_price to X
        adjusted_max_price = extracted_price
        
        # Allow some flexibility for high-rated products slightly above the price
        # This allows products with good reviews that are slightly more expensive
        flexibility_factor = 1.15  # Allow 15% above the specified price for high-rated products
        
    # Check if query contains price-related keywords without specific price
    has_price_keyword = any(keyword in q_norm for keyword in price_keywords)
    if has_price_keyword and not extracted_price and detected_category:
        # Set reasonable default price ranges based on category
        category_default_prices = {
            "mobile": 15000,
            "laptop": 40000,
            "shoes": 2000,
            "tshirts": 1500,
            "headphones": 5000,
            "watch": 15000,
            "accessories": 2000,
            "default": 5000
        }
        adjusted_max_price = category_default_prices.get(detected_category, category_default_prices["default"])
    
    # Call the enhanced search function with adjusted price parameters
    result = _compute_search(q_norm, adjusted_min_price, adjusted_max_price, min_rating, brand, category, sort)
    
    # Post-process results to include high-rated products slightly above price limit
    if extracted_price and result.get("results"):
        # Find products that are slightly above the price but have high ratings
        high_rated_above_limit = []
        price_limit = extracted_price
        flexibility_limit = price_limit * 1.15  # 15% flexibility
        
        for item in data:
            if (price_limit < item["price"] <= flexibility_limit and 
                item["rating"] >= 4.0 and 
                item.get("review_count", 0) >= 100):
                
                                 # Check if this item matches the query - be more strict
                 item_text = normalize(item["title"] + " " + item.get("description", ""))
                 item_category = normalize(item.get("category", ""))
                 
                 # Only include if the item's category matches the detected category
                 # For laptop category, check if item is actually a laptop
                 if detected_category == "laptop":
                     if any(keyword in item_text for keyword in ["laptop", "computer", "dell", "hp", "lenovo", "macbook"]):
                         high_rated_above_limit.append(item)
                 elif detected_category == "mobile":
                     if any(keyword in item_text for keyword in ["phone", "mobile", "smartphone", "iphone", "samsung", "oneplus", "vivo", "motorola"]):
                         high_rated_above_limit.append(item)
                 elif detected_category == "shoes":
                     if any(keyword in item_text for keyword in ["shoes", "footwear", "sneakers", "boots"]):
                         high_rated_above_limit.append(item)
                 elif detected_category == "tshirts":
                     if any(keyword in item_text for keyword in ["tshirt", "t-shirt", "shirt", "clothing"]):
                         high_rated_above_limit.append(item)
                 elif detected_category == "headphones":
                     if any(keyword in item_text for keyword in ["headphones", "earbuds", "airpods", "audio"]):
                         high_rated_above_limit.append(item)
                 elif detected_category == "watch":
                     if any(keyword in item_text for keyword in ["watch", "smartwatch", "apple watch"]):
                         high_rated_above_limit.append(item)
                 elif detected_category == "accessories":
                     if any(keyword in item_text for keyword in ["cover", "case", "mobile cover", "phone case"]):
                         high_rated_above_limit.append(item)
        
        # Add high-rated products to results (limit to 2-3 additional items)
        if high_rated_above_limit:
            # Sort by rating and review count
            high_rated_above_limit.sort(key=lambda x: (x["rating"], x.get("review_count", 0)), reverse=True)
            
            # Add to results (limit to 3 additional items)
            additional_items = high_rated_above_limit[:3]
            
            # Mark these as "high-rated alternatives"
            for item in additional_items:
                item["is_high_rated_alternative"] = True
            
            # Add to results
            result["results"].extend(additional_items)
            result["total_hits"] += len(additional_items)
    
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
#test commwnt