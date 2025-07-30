import json
import re

# Helper functions
def has_word(text, word):
    return re.search(rf"\\b{re.escape(word)}\\b", text, re.IGNORECASE) is not None

def list_has_word(lst, word):
    return any(has_word(str(x), word) for x in lst)

def normalize_tag(tag):
    return tag.strip().lower()

with open("backend/data/products.json", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    title = item.get("title", "")
    tags = set(item.get("tags", []))
    tags_norm = set(normalize_tag(t) for t in tags)
    subcategories = [str(x) for x in item.get("subcategories", [])]
    subcategories_norm = [normalize_tag(x) for x in subcategories]
    # --- Shirts ---
    if has_word(title, "shirt"):
        item["category_path"] = ["fashion", "clothing", "tops", "shirts"]
        is_women = has_word(title, "women") or "women" in tags_norm or "women" in subcategories_norm
        is_men = has_word(title, "men") or "men" in tags_norm or "men" in subcategories_norm
        if is_women:
            tags.add("Women")
        if is_men:
            tags.add("Men")
        if not is_women and not is_men:
            tags.update(["Men", "Women"])
        item["tags"] = sorted(set(tags), key=lambda x: x.lower())
    # --- Laptops ---
    if has_word(title, "laptop"):
        item["category_path"] = ["electronics", "computers", "laptops"]
        proc = None
        for p in ["i3", "i5", "i7", "i9"]:
            if has_word(title, p):
                proc = p
                break
        if "attributes" not in item or not isinstance(item["attributes"], dict):
            item["attributes"] = {}
        if proc:
            item["attributes"]["processor"] = proc
        # Remove any old Gaming/Business/General tags (case-insensitive)
        tags_norm = set(normalize_tag(t) for t in tags)
        tags = set(t for t in tags if normalize_tag(t) not in {"gaming", "business", "general"})
        # Add new tags as per rules
        added = False
        if has_word(title, "gaming") and "gaming" not in tags_norm:
            tags.add("Gaming")
            added = True
        if has_word(title, "business") and "business" not in tags_norm:
            tags.add("Business")
            added = True
        if not (has_word(title, "gaming") or has_word(title, "business")):
            tags.add("General")
        item["tags"] = sorted(set(tags), key=lambda x: x.lower())
    # --- Smartphones / Mobiles ---
    if has_word(title, "phone") or has_word(title, "mobile"):
        item["category_path"] = ["electronics", "mobile_phones", "smartphones"]
        tags_norm = set(normalize_tag(t) for t in tags)
        if "mobile" not in tags_norm:
            tags.add("Mobile")
        if "smartphone" not in tags_norm:
            tags.add("Smartphone")
        item["tags"] = sorted(set(tags), key=lambda x: x.lower())

with open("backend/data/products.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)