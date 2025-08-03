# Flipkart Search System

A full-stack search system inspired by Flipkart, featuring autosuggest, typo-tolerance, and semantic search over a real product dataset. Built with FastAPI (backend) and Next.js (frontend).

## Live Demo

**Frontend:** https://grid-front-deploy.vercel.app/  
**Backend:** https://grid-backend-20320365612.asia-south1.run.app/health

---

## Features

### 1. **Intelligent Autosuggestion with Typo-Tolerance**
Our advanced autosuggestion system handles spelling errors, phonetic matching, and fuzzy search to provide accurate suggestions even when users make typos.

<img width="1465" alt="Autosuggestion with typo-tolerance" src="https://github.com/user-attachments/assets/64a7c19f-a753-4134-b53e-7367592fd303">

### 2. **Context-Based Search**
The system remembers your search patterns and prioritizes relevant categories based on your previous searches. When you search for "shirt", the system saves the category path (Apparel→Men→Shirts→Casual Shirts) in local storage.

<img width="1469" alt="Context-based search - shirt category" src="https://github.com/user-attachments/assets/c4c5989c-1a58-4997-9c16-0a956c185e16">

Later, when you type "pan", pants-related products appear at the top of suggestions based on your browsing context.

<img width="1470" alt="Context-based search - pants suggestions" src="https://github.com/user-attachments/assets/874c68f6-9d6c-40e2-9e22-919c1505c461">

### 3. **Time-Based Smart Suggestions**
Our system adapts suggestions based on the time of day:
- **Early morning:** Typing "d" shows diya, dal, etc. (daily essentials)
- **Night time:** Typing "d" suggests daaru, wine, etc. (evening items)

### 4. **Delivery Days Filter**
A unique feature missing in many e-commerce sites - sort products by number of delivery days to find the fastest shipping options.

<img width="1470" alt="Sort by delivery days filter" src="https://github.com/user-attachments/assets/90ac3846-a06d-4592-9853-9bd1c9c253e6">

### 5. **Additional Features**
- **Semantic product search** using sentence transformers
- **Trending keywords** and multi-language support (Hindi-English ready)
- **Real Flipkart product data** (from Kaggle)
- **Modern, responsive frontend** with product grid, filters, and cart functionality

---

## Project Structure

```
flipkartSearchSystem/
├── backend/                # FastAPI backend
│   ├── data/
│   │   └── products.json   # Product database (JSON)
│   ├── main.py            # Main API server
│   └── requirements.txt   # Backend dependencies
├── frontend/              # Next.js frontend
│   ├── components/
│   ├── pages/
│   └── ...
└── dataset.csv           # (Optional) Raw product dataset
```

---

## Setup Instructions

### Quick Start (Recommended)
**Don't want to set up locally?** Use our deployed links:
- **Frontend:** https://grid-front-deploy.vercel.app/
- **Backend:** https://grid-backend-20320365612.asia-south1.run.app/health

### Local Development Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/prasaaaad98/Grid7.0.git
cd Grid7.0
```

#### 2. Backend Setup

**a. Create Virtual Environment (Recommended)**
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**b. Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**c. Run Backend Server**
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`

#### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
The frontend will be available at `http://localhost:3000`

> **Note:** If the frontend `.env` file contains `NEXT_PUBLIC_BACKEND_URL`, that URL will be used; otherwise, it defaults to `http://localhost:8000`

---

## Usage

1. Open `http://localhost:3000` in your browser
2. Use the search bar to test:
   - Autosuggest functionality
   - Typo-tolerance features
   - Semantic product search
3. Explore features:
   - Add products to cart
   - Apply filters
   - Sort by delivery days
   - Experience context-based suggestions

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** this repository and clone your fork
2. **Create a new branch** for your feature or bugfix
3. **Make your changes** (backend or frontend)
4. **Test thoroughly** (run both backend and frontend locally)
5. **Submit a pull request** with a clear description

### Ideas for Contribution

- Improve autosuggest ranking algorithms
- Add more filters (brand, category, price range)
- Enhance product detail pages
- Implement user authentication and order history
- Optimize backend for larger datasets
- Add Hindi-English translation for queries
- Implement real-time inventory updates
- Add recommendation system based on user behavior

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend slow to start | Reduce dataset size in `convert_csv_to_json.py` |
| Unicode errors | Ensure all files use `encoding='utf-8'` |
| Dependency errors | Check Python and Node.js versions |
| Frontend not connecting to backend | Verify `NEXT_PUBLIC_BACKEND_URL` in `.env` |

---

## Data Setup (Optional)

The backend uses `backend/data/products.json` as its database. To use a new dataset:

1. Place your CSV as `dataset.csv` in the project root
2. Run the conversion script:
   ```bash
   python convert_csv_to_json.py
   ```
3. This generates `backend/data/products.json` in the correct format

---

## License

MIT License - feel free to use this project for learning and development.

---

## Credits

- **Data Scraping:** [EcommerceAPI.io](https://api.ecommerceapi.io/)
- **Product Dataset:** [Kaggle Flipkart Product Dataset](https://www.kaggle.com/datasets/priyankkhanna/flipkart-product-dataset-by-priyank-khanna)
- **Built with:** FastAPI, Next.js, and open-source libraries
- **Created by:** Grid 7.0 Team

---

**Star this Repository**

If you found this project helpful, please consider giving it a star!

---

*Happy Searching!
