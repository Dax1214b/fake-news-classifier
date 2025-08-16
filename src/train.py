import argparse
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils import load_data

DEFAULT_DATA = str(Path(__file__).resolve().parents[1] / "data" / "sample.csv")
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "model.joblib")

def train(data_path: str = DEFAULT_DATA, model_out: str = MODEL_PATH):
    df = load_data(data_path)
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
    ])
    pipe.fit(X, y)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to CSV with columns: text,label")
    ap.add_argument("--out", type=str, default=MODEL_PATH, help="Where to save the trained model")
    args = ap.parse_args()
    train(args.data, args.out)
