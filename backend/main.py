from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel, Field
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
import pygtrie

# ------------------------- Setup -------------------------
data = json.load(open("data/products.json", encoding='utf-8'))

# Transform data to align with new schema
for item in data:
    # Rename ratings to rating
    if 'ratings' in item:
        item['rating'] = item.pop('ratings')
    
    # Add missing fields with defaults
    if 'images' not in item:
        item['images'] = []
    if 'retail_price' not in item:
        item['retail_price'] = item.get('price', 0.0)
    if 'subcategories' not in item:
        item['subcategories'] = []

# Define hierarchical categories and keywords
category_rules = [
    ("electronics/heater/water heater/aquarium heater", ["aquarium heater"]),
    ("electronics/heater/water heater", ["water heater", "heater"]),
    ("electronics/heater", ["heater"]),
    ("electronics/smartphone", ["smartphone", "phone", "mobile"]),
    ("electronics/headphones", ["headphones", "earbuds", "earphones"]),
    ("sports/cricket/bat", ["cricket bat", "bat"]),
    ("sports/cricket/ball", ["cricket ball", "ball"]),
    ("sports/cricket/net", ["cricket net", "net"]),
    ("sports/badminton/racquet", ["badminton racquet", "racquet"]),
    ("sports/badminton/shuttlecock", ["shuttlecock", "shuttle"]),    
    ("other", [])
]

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def categorize_product(title: str) -> str:
    text = title.lower()
    for category, keywords in category_rules:
        for kw in keywords:
            if kw in text:
                return category
    return "other"

# Assign category hierarchy and top-level category
for item in data:
    hierarchy = categorize_product(item.get('title',""))
    item['category_hierarchy'] = hierarchy
    item['category'] = hierarchy.split('/')[0] if '/' in hierarchy else hierarchy

product_titles = [item['title'] for item in data]

nlp = spacy.load("en_core_web_sm")

# Translation helper (stub) and normalization

def translate_hindi_to_english(text: str) -> str:
    return normalize(text)

# Phrase extraction for suggestions
def extract_phrases(title: str):
    doc = nlp(title)
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = normalize(chunk.text)
        if 1 <= len(chunk.text.split()) <= 4 and len(phrase) >= 2:
            phrases.append(phrase)
    return phrases

# Build suggestion phrases set
all_phrases = set()
for title in product_titles:
    all_phrases.update(extract_phrases(title))
for title in product_titles:
    for token in normalize(title).split():
        if len(token) > 1:
            all_phrases.add(token)
suggestion_phrases = sorted(all_phrases)

# Trie for prefix matching
phrase_trie = pygtrie.CharTrie()
for phrase in suggestion_phrases:
    phrase_trie[phrase] = None

# Phonetics map
phrase_phonetics = {phrase: doublemetaphone(phrase)[0] for phrase in suggestion_phrases}
product_titles_norm = [normalize(t) for t in product_titles]
product_titles_phonetic = [doublemetaphone(normalize(t))[0] for t in product_titles]

# Hardcoded trending product titles map per alphabet
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

# Create trending keywords set from trending_map
trending_keywords = set()
for letter_products in trending_map.values():
    trending_keywords.update([normalize(product) for product in letter_products])

# Semantic model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(product_titles)
dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

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
    brand: str
    category: str
    price: float
    retail_price: Optional[float] = None
    images: List[str] = []
    rating: float
    description: str
    category_hierarchy: str
    subcategories: List[str] = []
    
    class Config:
        allow_population_by_field_name = True

@app.get("/autosuggest", response_model=List[str])
def autosuggest(q: str = Query(..., min_length=1)):
    q_norm = normalize(q)
    q_translated = translate_hindi_to_english(q)
    if q_translated != q_norm:
        q_norm = q_translated

    # For single letter queries, return only trending products
    if len(q_norm) == 1:
        return trending_map.get(q_norm, [])
    
    # For queries with 2 or more characters, use all algorithms
    suggestions = []
    
    # Trending products that start with the prefix
    suggestions = [t for t in trending_map.get(q_norm[0], []) if normalize(t).startswith(q_norm)]
    
    # Trie-based prefix matches
    starts = []
    try:
        starts = list(phrase_trie.keys(prefix=q_norm))
    except Exception:
        starts = []
    suggestions.extend([p for p in starts if p not in suggestions])
    
    # Fuzzy matching
    fuzzy = process.extract(q_norm, suggestion_phrases, scorer=(fuzz.token_set_ratio if ' ' in q_norm else fuzz.partial_ratio), limit=10)
    suggestions.extend([m[0] for m in fuzzy if m[1]>=60 and m[0] not in suggestions])
    
    # Substring matching
    suggestions.extend([p for p in suggestion_phrases if q_norm in p and p not in suggestions])
    
    # Phonetic matching
    code = doublemetaphone(q_norm)[0]
    suggestions.extend([p for p,c in phrase_phonetics.items() if c==code and p not in suggestions])
    
    # Dedupe & return top 5
    res = []
    for s in suggestions:
        if s not in res and len(s) > 1:
            res.append(s)
    
    return res[:5]

@app.get("/search", response_model=List[Product])
def search(
    q: str = Query(..., min_length=1),
    min_price: float = 0,
    max_price: float = 1e6,
    min_rating: float = 0,
    brand: str = "",
    category: str = ""
):
    q_norm = normalize(q)
    q_translated = translate_hindi_to_english(q)
    if q_translated!=q_norm:
        q_norm = q_translated
    # NER for price/brand/category
    doc = nlp(q)
    for ent in doc.ents:
        if ent.label_=="MONEY":
            num = re.sub(r"\D","",ent.text)
            if num.isdigit(): max_price=float(num)
        if ent.label_=="ORG": brand=ent.text.lower()
        if ent.label_ in ["PRODUCT","NORP"]: category=ent.text.lower()
    # Typo correction
    best = process.extractOne(q_norm, product_titles_norm, scorer=fuzz.partial_ratio)
    if best and best[1]>=70:
        q_norm = normalize(best[0])
    else:
        code = doublemetaphone(q_norm)[0]
        for i,c in enumerate(product_titles_phonetic):
            if c==code:
                q_norm = normalize(product_titles[i])
                break
    # Semantic search
    vec = model.encode([q_norm])
    _, idx = index.search(np.array(vec),50)
    candidates=[data[i] for i in idx[0]]
    # Filters
    filtered=[]
    for item in candidates:
        if not(min_price<=item.get('price', 0)<=max_price): continue
        if item.get('rating', 0)<min_rating: continue
        if brand and brand not in normalize(item.get('title', '')): continue
        if category and category not in normalize(item.get('title', '')): continue
        filtered.append(item)
    # Rank
    def score_item(item):
        tv=model.encode([item.get('title', '')])[0]
        sim=np.dot(vec,tv)/(np.linalg.norm(vec)*np.linalg.norm(tv))
        rating_score = item.get('rating', 0) / 5
        trending_bonus = 0.2 if normalize(item.get('title', '')) in trending_keywords else 0
        return 0.5*sim + 0.3*rating_score + trending_bonus
    ranked=sorted(filtered, key=score_item, reverse=True)
    # Response
    return [
        {**item, 'category_hierarchy': item.get('category_hierarchy', 'other')}
        for item in ranked[:10]
    ]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
