# Flipkart Search System

A full-stack search system inspired by Flipkart, featuring autosuggest, typo-tolerance, and semantic search over a real product dataset. Built with FastAPI (backend) and Next.js (frontend).

---

## Features
- **Autosuggest** with typo-tolerance, phonetic, and fuzzy matching
- **Semantic product search** using sentence transformers
- **Trending keywords** and multi-language support (Hindi-English ready)
- **Real Flipkart product data** (from Kaggle)
- **Modern, responsive frontend** with product grid, filters, and cart

---

## Project Structure
```
flipkartSearchSystem/
  backend/           # FastAPI backend
    data/products.json  # Product database (JSON)
    main.py             # Main API server
    requirements.txt    # Backend dependencies
  frontend/          # Next.js frontend
    ...
  dataset.csv        # (Optional) Raw product dataset (CSV)
```

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <repo-url>
cd flipkartSearchSystem
```

### 2. Backend Setup
#### a. Create and activate a virtual environment (optional but recommended)
```sh
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

#### b. Install dependencies
```sh
pip install -r requirements.txt
```
b.1 After above installations, run this as well
```sh
python -m spacy download en_core_web_sm
```

#### c. Prepare the product data
- The backend uses `backend/data/products.json` as its database.
- To use a new dataset (e.g., from Kaggle):
  1. Place your CSV as `dataset.csv` in the project root.
  2. Run the conversion script:
     ```sh
     python convert_csv_to_json.py
     ```
  3. This will generate `backend/data/products.json` in the correct format.

#### d. Run the backend server
```sh
uvicorn main:app --reload
```
- The API will be available at `http://localhost:8000`

### 3. Frontend Setup
```sh
cd ../frontend
npm install
npm run dev
```
- The frontend will be available at `http://localhost:3000`

---

## Usage
- Open `http://localhost:3000` in your browser.
- Use the search bar to try autosuggest, typo-tolerance, and product search.
- Add products to cart, filter, and explore the UI.

---

## Contributing
1. **Fork this repo** and clone your fork.
2. **Create a new branch** for your feature or bugfix.
3. **Make your changes** (backend or frontend).
4. **Test thoroughly** (run both backend and frontend locally).
5. **Submit a pull request** with a clear description of your changes.

### Ideas for Contribution
- Improve autosuggest ranking or add new matching strategies
- Add more filters (brand, category, price, etc.)
- Enhance product detail pages
- Add user authentication or order history
- Optimize backend for larger datasets
- Add Hindi-English translation for queries

---

## Troubleshooting
- If the backend is slow to start, try reducing the dataset size in `convert_csv_to_json.py`.
- For Unicode errors, ensure all files are read/written with `encoding='utf-8'`.
- If you see dependency errors, check your Python and Node.js versions.

---

## License
MIT (or specify your license here)

---

## Credits
- Product data: [Kaggle Flipkart Product Dataset](https://www.kaggle.com/datasets/priyankkhanna/flipkart-product-dataset-by-priyank-khanna)
- Built with FastAPI, Next.js, and open-source libraries 
