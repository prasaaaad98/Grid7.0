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
        # Extract the required fields using the actual column names
        product = {
            "id": index + 1,  # Generate sequential IDs
            "title": str(row.get('title', f'Product {index + 1}')),
            "price": clean_price(row.get('selling_price')),
            "image": str(row.get('image_links', f'https://dummyimage.com/product{index + 1}')),
            "rating": float(row.get('product_rating', random.uniform(3.5, 5.0))),
            "description": str(row.get('description', ''))
        }
        
        # Clean up the data
        if product['title'] == 'nan' or product['title'] == '':
            product['title'] = f'Product {index + 1}'
        
        if product['price'] <= 0 or pd.isna(product['price']):
            product['price'] = random.randint(100, 50000)
            
        if product['rating'] <= 0 or pd.isna(product['rating']):
            product['rating'] = round(random.uniform(3.5, 5.0), 1)
            
        if product['image'] == 'nan' or product['image'] == '':
            product['image'] = f'https://dummyimage.com/product{index + 1}'
            
        if product['description'] == 'nan' or product['description'] == '':
            product['description'] = f'High quality {product["title"]} with excellent features'
        
        products.append(product)
    
    # Save to JSON file
    output_file = 'backend/data/products.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(products)} products to {output_file}")
    print(f"Sample product: {products[0] if products else 'No products found'}")

if __name__ == "__main__":
    convert_csv_to_json() 