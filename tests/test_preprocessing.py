# tests/test_preprocessing.py
from pathlib import Path
import pandas as pd

def test_splits_exist_and_scaled():
    base = Path("artifacts/splits")
    for name in ["train.csv", "val.csv", "test.csv"]:
        p = base / name
        assert p.exists(), f"missing {p}"
        df = pd.read_csv(p)
        assert "label" in df.columns
        px_cols = [c for c in df.columns if c.startswith("px")]
        assert len(px_cols) == 784

        X = df[px_cols]
        assert X.min().min() >= -1e-6
        assert X.max().max() <= 1.0 + 1e-6
