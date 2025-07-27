import pandas as pd
import json
import random
import re

def clean_price(price_str):
    """Clean price string by removing currency symbols and commas"""
    if pd.isna(price_str) or price_str == '':
        return random.randint(100, 50000)
    
    # Convert to string and remove currency symbols, commas, and spaces
    price_str = str(price_str)
    price_str = re.sub(r'[₹$,\s]', '', price_str)
    
    try:
        return float(price_str)
    except ValueError:
        return random.randint(100, 50000)

def convert_csv_to_json():
    # Read the CSV file
    print("Reading dataset.csv...")
    df = pd.read_csv('dataset.csv')
    
    print(f"Found {len(df)} products in the dataset")
    print("Columns available:", df.columns.tolist())
    
    # Create the products list with the required structure
    products = []
    
    for index, row in df.iterrows():
        # Parse images as array
        images = []
        image_field = row.get('image', '')
        if isinstance(image_field, str) and image_field.strip().startswith('['):
            try:
                # Remove extra quotes and parse as list
                images = eval(image_field)
                if not isinstance(images, list):
                    images = [str(image_field)]
            except Exception:
                images = [str(image_field)]
        elif isinstance(image_field, str) and image_field:
            images = [image_field]
        else:
            images = [f'https://dummyimage.com/product{index + 1}']

        # Clean category
        category = row.get('product_category_tree', '')
        if isinstance(category, str) and category.strip().startswith('['):
            try:
                # Remove extra quotes and parse as list, take the last category as most specific
                cat_list = eval(category)
                if isinstance(cat_list, list) and len(cat_list) > 0:
                    category = cat_list[-1].split('>>')[-1].strip()
                else:
                    category = 'General'
            except Exception:
                category = 'General'
        elif isinstance(category, str) and category:
            category = category
        else:
            category = 'General'

        # Clean price fields
        price = clean_price(row.get('discounted_price'))
        retail_price = clean_price(row.get('retail_price'))

        # Clean rating
        try:
            rating = float(row.get('product_rating'))
            if rating <= 0 or pd.isna(rating):
                rating = round(random.uniform(3.5, 5.0), 1)
        except Exception:
            rating = round(random.uniform(3.5, 5.0), 1)

        # Build product dict
        product = {
            "id": index + 1,
            "title": str(row.get('product_name', f'Product {index + 1}')),
            "brand": str(row.get('brand', 'Unknown')),
            "category": category,
            "price": price,
            "retail_price": retail_price,
            "images": images,
            "rating": rating,
            "description": str(row.get('description', f'High quality {row.get('product_name', f'Product {index + 1}')}')),
        }

        # Fallbacks for missing/invalid values
        if not product['title'] or product['title'] == 'nan':
            product['title'] = f'Product {index + 1}'
        if not product['brand'] or product['brand'] == 'nan':
            product['brand'] = 'Unknown'
        if not product['category'] or product['category'] == 'nan':
            product['category'] = 'General'
        if not product['images'] or product['images'][0] == 'nan':
            product['images'] = [f'https://dummyimage.com/product{index + 1}']
        if not product['description'] or product['description'] == 'nan':
            product['description'] = f'High quality {product["title"]}'
        if not product['price'] or pd.isna(product['price']) or product['price'] <= 0:
            product['price'] = random.randint(100, 50000)
        if not product['retail_price'] or pd.isna(product['retail_price']) or product['retail_price'] <= 0:
            product['retail_price'] = product['price'] + random.randint(100, 1000)
        if not product['rating'] or pd.isna(product['rating']) or product['rating'] <= 0:
            product['rating'] = round(random.uniform(3.5, 5.0), 1)

        products.append(product)
    
    # Save to JSON file
    output_file = 'backend/data/products.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(products)} products to {output_file}")
    print(f"Sample product: {products[0] if products else 'No products found'}")

if __name__ == "__main__":
    convert_csv_to_json() 