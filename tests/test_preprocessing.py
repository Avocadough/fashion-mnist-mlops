import pandas as pd
from pathlib import Path

def test_splits_exist_and_scaled():
    base = Path("artifacts/splits")
    for name in ["train.csv", "val.csv", "test.csv"]:
        p = base / name
        assert p.exists()
        df = pd.read_csv(p)
        # ต้องมีคอลัมน์ 784 + label
        assert "label" in df.columns
        assert len([c for c in df.columns if c.startswith("px")]) == 784
        # เช็คสเกล 0..1
        X = df.drop(columns=["label"])
        assert X.min().min() >= 0.0 - 1e-6
        assert X.max().max() <= 1.0 + 1e-6
