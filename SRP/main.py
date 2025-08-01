from fastapi import FastAPI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json
app = FastAPI()
es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed(text: str):
    return model.encode(text).tolist()

@app.get("/search/")
def search_products(keywords: str, page: int = 1, size: int = 20):
    from_ = (page - 1) * size
    query_vector = embed(keywords) 
    search_query = {
        "from": from_,
        "size": size,
        "_source": ["title", "description", "price", "brand"],
        "query": {
            "match": {
                "description": {
                    "query": keywords,
                    "boost": 0.5
                }
            }
        },
        "rescore": {
            "window_size": 100,
            "query": {
                "rescore_query": {
                    "sltr": {
                        "params": {
                            "keywords": keywords,
                            "query_vector": query_vector
                        },
                        "model": "smartsearch_linear_model",
                        "featureset": "smartsearch_ltr_features"
                    }
                }
            }
        }
    }
    print(json.dumps(search_query, indent=2))
    response = es.search(index="products", body=search_query)
    results = [hit["_source"] for hit in response["hits"]["hits"]]

    return {
        "total_hits": response["hits"]["total"]["value"],
        "page": page,
        "results": results
    }
