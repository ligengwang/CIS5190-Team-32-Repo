# CIS5190-Team-32-Repo

Link to our latex report: https://www.overleaf.com/project/69dee6a89637524f6583b00b

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/lesleyzhao/CIS5190-Team-32-Repo.git
cd CIS5190-Team-32-Repo
```

### 2. Set Up Virtual Environment on VS Code
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install pandas torch nltk scikit-learn transformers
```

### 4. Run Preprocessing
Cleans the raw scraped headlines and saves a model-ready CSV:
```bash
python3 preprocess.py
```
Output: `scraped_headlines_clean_latest.csv`

### 5. Run Baseline Model
```bash
python3 tfidf_baseline.py
```

### 6. Run Improved Models
```bash
python3 model.py
python3 train.py
python3 roberta.py
python3 train_roberta.py
```

## Project Structure
To be continued..