# scripts/3_model_training.py
# Train several runs with PCA -> LogisticRegression (multinomial).
import os, json, pathlib
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
ARTIFACTS = pathlib.Path("artifacts")
SPLITS = ARTIFACTS / "splits"
CAND_PATH = ARTIFACTS / "candidates.json"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fashion_mnist_coursework")

# --- Force MLflow to local file store (no server / no proxy) ---
for k in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACTS_URI", "MLFLOW_ENABLE_ARTIFACTS_PROXY"]:
    os.environ.pop(k, None)
mlflow.set_tracking_uri(pathlib.Path("mlruns").resolve().as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)
# ---------------------------------------------------------------

def main():
    # Read splits with compact dtypes to save memory
    pixel_dtypes = {f"px{i}": "float32" for i in range(784)}
    dtypes = {**pixel_dtypes, "label": "int8"}

    train = pd.read_csv(SPLITS / "train.csv", dtype=dtypes)
    val   = pd.read_csv(SPLITS / "val.csv",   dtype=dtypes)

    Xtr, ytr = train.drop(columns=["label"]).values, train["label"].values
    Xva, yva = val.drop(columns=["label"]).values,   val["label"].values

    candidates = []
    # Try a couple of PCA sizes + C values (fast enough)
    for n_comp in [50, 100]:
        for C in [0.5, 1.0, 2.0]:
            run_name = f"3_train_PCA{n_comp}_C{C}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model": "LogisticRegression",
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "multiclass": "multinomial",
                    "pca_components": n_comp,
                    "C": C,
                    "seed": SEED,
                })

                pipe = Pipeline([
                    ("pca", PCA(n_components=n_comp, random_state=SEED)),
                    ("clf", LogisticRegression(
                        C=C, max_iter=1000, random_state=SEED,
                        multi_class="multinomial"  # lbfgs
                    ))
                ])

                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xva)
                acc = accuracy_score(yva, pred)
                f1  = f1_score(yva, pred, average="macro")
                mlflow.log_metrics({"val_accuracy": float(acc), "val_f1_macro": float(f1)})

                cm = confusion_matrix(yva, pred)
                mlflow.log_text(np.array2string(cm), "reports/confusion_matrix_val.txt")

                # Log the model artifact for this run
                mlflow.sklearn.log_model(pipe, artifact_path="model")

                run_id = mlflow.active_run().info.run_id
                candidates.append({
                    "run_id": run_id,
                    "pca_components": n_comp,
                    "C": C,
                    "val_accuracy": float(acc),
                    "val_f1_macro": float(f1)
                })
                print(f"[OK] {run_name}: acc={acc:.4f}, f1={f1:.4f}, run_id={run_id}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(CAND_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    print("[OK] Training finished. Candidates saved to", CAND_PATH)

if __name__ == "__main__":
    main()
