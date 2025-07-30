import json
import re

def has_word(text, word):
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None

def list_has_word(lst, word):
    return any(has_word(str(x), word) for x in lst)

with open("backend/data/products.json", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    title = item.get("title", "")
    tags = set(item.get("tags", []))
    subcategories = [str(x) for x in item.get("subcategories", [])]
    # --- Shirts ---
    if has_word(title, "shirt"):
        # Set category_path
        item["category_path"] = ["fashion", "clothing", "tops", "shirts"]
        # Women/Men detection
        is_women = has_word(title, "women") or list_has_word(tags, "women") or list_has_word(subcategories, "Women")
        is_men = has_word(title, "men") or list_has_word(tags, "men") or list_has_word(subcategories, "Men")
        if is_women:
            tags.add("Women")
        if is_men:
            tags.add("Men")
        if not is_women and not is_men:
            tags.update(["Men", "Women"])
        item["tags"] = list(tags)
    # --- Laptops ---
    if has_word(title, "laptop"):
        item["category_path"] = ["electronics", "computers", "laptops"]
        # Processor detection
        proc = None
        for p in ["i3", "i5", "i7", "i9"]:
            if has_word(title, p):
                proc = p
                break
        if proc:
            if "attributes" not in item or not isinstance(item["attributes"], dict):
                item["attributes"] = {}
            item["attributes"]["processor"] = proc
        # Gaming/Business/General tags
        if has_word(title, "gaming"):
            tags.add("Gaming")
        if has_word(title, "business"):
            tags.add("Business")
        if not (has_word(title, "gaming") or has_word(title, "business")):
            tags.add("General")
        item["tags"] = list(tags)
    # --- Smartphones / Mobiles ---
    if has_word(title, "phone") or has_word(title, "mobile"):
        item["category_path"] = ["electronics", "mobile_phones", "smartphones"]
        tags.update(["Mobile", "Smartphone"])
        item["tags"] = list(tags)

with open("backend/data/products.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)