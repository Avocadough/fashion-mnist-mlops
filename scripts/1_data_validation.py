# scripts/1_data_validation.py
# Fashion-MNIST (70,000 rows): download once via OpenML (cached next runs)
import os, pathlib
import pandas as pd
import mlflow
from sklearn.datasets import fetch_openml

# --- force local mlruns (no server / no proxy) ---
for k in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACTS_URI", "MLFLOW_ENABLE_ARTIFACTS_PROXY"]:
    os.environ.pop(k, None)
mlflow.set_tracking_uri(pathlib.Path("mlruns").resolve().as_uri())
mlflow.set_experiment("fashion_mnist_coursework")
# --------------------------------------------------

ARTIFACTS = pathlib.Path("artifacts")

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False)  # ~70MB first time
    X = pd.DataFrame(fm.data, columns=[f"px{i}" for i in range(fm.data.shape[1])])
    y = pd.Series(fm.target.astype(int), name="label")
    df = pd.concat([X, y], axis=1)
    csv_path = ARTIFACTS / "fashion_mnist.csv"
    df.to_csv(csv_path, index=False)

    with mlflow.start_run(run_name="1_data_validation_fashion"):
        mlflow.log_param("dataset", "fashion_mnist")
        mlflow.log_metric("num_rows", df.shape[0])
        mlflow.log_metric("num_cols", df.shape[1])
        mlflow.log_metric("missing_values", int(df.isna().sum().sum()))
        mlflow.log_param("num_classes", int(df["label"].nunique()))
        mlflow.log_artifact(str(csv_path))

        assert df.shape[0] >= 60000, "Dataset rows too few"
        assert df["label"].nunique() >= 5, "Too few classes"
    print("[OK] fashion_mnist.csv saved with shape:", df.shape)

if __name__ == "__main__":
    main()
