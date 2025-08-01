from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "products"

def normalize_booleans(product):
    """Converts 'Yes'/'No' strings to boolean True/False values."""
    if "attributes" in product:
        for key, val in product["attributes"].items():
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered == "yes":
                    product["attributes"][key] = True
                elif lowered == "no":
                    product["attributes"][key] = False

    for key, val in product.items():
        if isinstance(val, str):
            lowered = val.strip().lower()
            if lowered == "yes":
                product[key] = True
            elif lowered == "no":
                product[key] = False

    return product

# Delete existing index
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

# Create index with dynamic attributes and vector search
mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "integer"},
            "brand": {"type": "keyword"},
            "title": {"type": "text"},
            "category": {"type": "keyword"},
            "description": {"type": "text"},
            "subcategories": {"type": "keyword"},
            "category_path": {"type": "keyword"},
            "synonyms": {"type": "keyword"},
            "attributes": {
                "type": "object",
                "dynamic": True
            },
            "tags": {"type": "keyword"},
            "related_ids": {"type": "integer"},
            "price": {"type": "float"},
            "retail_price": {"type": "float"},
            "review_count": {"type": "integer"},
            "images": {"type": "keyword"},
            "rating": {"type": "float"},
            "best_time_to_buy": {"type": "keyword"},
            "stock": {"type": "integer"},
            "how_many_purchased": {"type": "integer"},
            "is_best_seller": {"type": "boolean"},
            "contextual_data": {
                "properties": {
                    "ideal_for": {"type": "text"},
                    "material": {"type": "keyword"},
                    "origin": {"type": "keyword"}
                }
            },
            "search_boost": {"type": "float"},
            "color": {"type": "keyword"},
            "material": {"type": "keyword"},
            "finish": {"type": "keyword"},
            "plus_eligible": {"type": "boolean"},
            "delivery_time": {"type": "integer"},
            "offers": {"type": "keyword"},
            "exchange_available": {"type": "boolean"},
            "assured_badge": {"type": "boolean"},
            "warehouse_loc": {
                "properties": {
                    "name": {"type": "keyword"},
                    "coords": {"type": "geo_point"}
                }
            },
            "title_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },

            "description_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

es.indices.create(index=INDEX_NAME, body=mapping)
print("Index created.")

# Load product data
with open("final_data.json", "r", encoding="utf-8") as f:
    products = json.load(f)

# Process and index products
for product in products:
    description = product.get("description", "")
    description_embedding = model.encode(description, normalize_embeddings=True).tolist()
    product["description_vector"] = description_embedding

    title = product.get("title", "")
    title_embedding = model.encode(title, normalize_embeddings=True).tolist()
    product["title_vector"] = title_embedding

    product = normalize_booleans(product)
    es.index(index=INDEX_NAME, document=product)

print(f"Indexed {len(products)} products.")
