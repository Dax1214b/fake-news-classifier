import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_data

DEFAULT_DATA = str(Path(__file__).resolve().parents[1] / "data" / "sample.csv")
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "model.joblib")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

def evaluate(data_path: str = DEFAULT_DATA, model_path: str = MODEL_PATH):
    df = load_data(data_path)
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, digits=4)
    cm = confusion_matrix(y, y_pred, labels=["FAKE", "REAL"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}")

    # Plot confusion matrix
    plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0,1]
    plt.xticks(tick_marks, ["FAKE","REAL"])
    plt.yticks(tick_marks, ["FAKE","REAL"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200)
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print(f"Saved metrics to {REPORT_DIR / 'metrics.txt'} and confusion matrix plot.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to CSV with columns: text,label")
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained model")
    args = ap.parse_args()
    evaluate(args.data, args.model)
