# Split + scale pixels to [0,1] and (optionally) save a small preview image grid as CSV only (no images on disk)
import pathlib, pandas as pd
from sklearn.model_selection import train_test_split

ARTIFACTS = pathlib.Path("artifacts")
SPLITS = ARTIFACTS / "splits"
SEED = 42

def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ARTIFACTS / "fashion_mnist.csv")

    # scale 0..255 -> 0..1
    X = df.drop(columns=["label"]).astype("float32") / 255.0
    y = df["label"].astype(int)

    # train/val/test = 50k/10k/10k (approx)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    pd.concat([X_train, y_train], axis=1).to_csv(SPLITS / "train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(SPLITS / "val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(SPLITS / "test.csv", index=False)

    print("[OK] Preprocessing complete. Splits saved to", SPLITS)

if __name__ == "__main__":
    main()
