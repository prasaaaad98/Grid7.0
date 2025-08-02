# 🔎 Learning-to-Rank Elasticsearch Setup (v9.0.1)

This guide helps you set up Elasticsearch 9.0.1 with the [LTR (Learning to Rank)](https://github.com/o19s/elasticsearch-learning-to-rank) plugin, seed product data, and upload the required feature set and model.

---

## 📦 Step 1: Download and Extract Elasticsearch 9.0.1

### 🔗 Download Link:
[https://www.elastic.co/downloads/past-releases/elasticsearch-9-0-1](https://www.elastic.co/downloads/past-releases/elasticsearch-9-0-1)

### 📂 Extract

#### On Linux/macOS:
tar -xzf elasticsearch-9.0.1-linux-x86_64.tar.gz
cd elasticsearch-9.0.1
On Windows (PowerShell or CMD):
Use any archive tool (like 7-Zip) to extract elasticsearch-9.0.1-windows-x86_64.zip

Rename the extracted folder to elasticsearch-9.0.1

Navigate into the folder in Command Prompt or PowerShell

🔌 Step 2: Install LTR Plugin
📥 Download Plugin
Go to the LTR GitHub Releases:
https://github.com/o19s/elasticsearch-learning-to-rank/releases
Download the ZIP for version compatible with Elasticsearch 9.0.1.

📁 Install Plugin
Inside elasticsearch-9.0.1/, create the plugin folder:

plugins/ltr/
Move the contents of the unzipped LTR plugin folder into plugins/ltr.

Final structure should look like:
elasticsearch-9.0.1/
└── plugins/
    └── ltr/
        ├── plugin-descriptor.properties
        └── ...



▶️ Step 3: Run Elasticsearch
On Linux/macOS:
./bin/elasticsearch
On Windows (PowerShell or CMD):
.\bin\elasticsearch.bat
Wait until Elasticsearch is fully started (usually on port 9200).

🐍 Step 4: Install Python Requirements
Ensure you're in the project root directory.

All OS:
pip install -r requirements.txt


🌱 Step 5: Seed Data
Run the script to index products into Elasticsearch.

On Linux/macOS:
python seed_data.py

On Windows:
python .\seed_data.py

📤 Step 6: Upload FeatureSet and Model
✅ Upload FeatureSet
Linux/macOS:
curl -XPOST -H "Content-Type: application/json" ^
  --data-binary @featureset.json ^
  http://localhost:9200/_ltr/_featureset/smartsearch_ltr_features


Windows (CMD or PowerShell):
curl -X POST "http://localhost:9200/_ltr/_featureset/smartsearch_ltr_features" -H "Content-Type: application/json" --data-binary "@featureset.json"


✅ Upload Model
Linux/macOS:
curl -XPOST -H "Content-Type: application/json" ^
  --data-binary @model.json ^
  http://localhost:9200/_ltr/_model/smartsearch_linear_model
Windows (CMD or PowerShell):
curl -X POST "http://localhost:9200/_ltr/_model/smartsearch_linear_model" -H "Content-Type: application/json"  --data-binary "@model.json"

  
✅ Done!
Your Elasticsearch LTR setup is ready. You can now use LTR in your search pipeline to re-rank search results.

📁 Project Structure
bash
Copy
Edit
.
├── featureset.json          # LTR feature definitions
├── model.json               # LTR model (e.g. linear model)
├── final_data.json          # Data used for seeding
├── seed_data.py             # Python script to seed Elasticsearch
├── main.py                  # (Optional) main script to test pipeline
├── requirements.txt         # Python dependencies
└── README.md                # You're reading it!

🛠️ Troubleshooting
Make sure the LTR plugin matches the exact Elasticsearch version.

Elasticsearch may take a minute to fully start. Wait for “started” message in logs.

Use http://localhost:9200/_cat/indices?v to verify your index and documents.
