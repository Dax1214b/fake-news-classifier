from pathlib import Path
import pandas as pd

def load_data(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(p)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df
