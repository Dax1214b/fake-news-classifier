# Fake News Classifier

A clean, interview-ready ML project that predicts whether a news headline/text is **REAL** or **FAKE** using TF‑IDF + Logistic Regression.

## 🚀 Quick Start

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train the model (uses sample data by default)
python src/train.py

# 4) Run the Streamlit app
streamlit run app.py
```

The app lets you paste any headline or short news text and get a prediction with model confidence.

## 📂 Project Structure

```
fake-news-classifier/
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
├── README.md              # This file
├── src/
│   ├── train.py          # Training script (saves pipeline)
│   ├── eval.py           # Evaluation script (metrics + confusion matrix)
│   └── utils.py          # Shared helpers
├── data/
│   └── sample.csv        # Tiny dataset for quick demo
├── models/
│   └── model.joblib      # Saved sklearn Pipeline (created after training)
└── reports/
    ├── metrics.txt       # Metrics summary
    └── confusion_matrix.png
```

## 🧠 Model & Approach

- **Vectorizer:** `TfidfVectorizer` with basic English stopwords
- **Classifier:** `LogisticRegression`
- **Target labels:** `REAL`, `FAKE`

Everything is wrapped in a single scikit-learn **Pipeline** for reproducibility.

## 🔁 Using a Larger Dataset (Optional but Recommended)

Replace `data/sample.csv` with a larger dataset that has **two columns**:
- `text`: the news headline or short article
- `label`: either `REAL` or `FAKE`

Then re-run training:
```bash
python src/train.py --data data/your_dataset.csv
python src/eval.py  --data data/your_dataset.csv
```

## 📝 License

MIT
