# scripts/4_model_evaluation.py
# Pick best by val_accuracy; evaluate on test; save model folder for serving
import os, json, pathlib, shutil
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
ARTIFACTS = pathlib.Path("artifacts")
SPLITS = ARTIFACTS / "splits"
CAND_PATH = ARTIFACTS / "candidates.json"
BEST_PATH = ARTIFACTS / "best_run.json"
DEPLOY_DIR = ARTIFACTS / "deploy" / "fashion_mnist_clf"

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fashion_mnist_coursework")
THRESH = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))  # Fashion-MNIST with PCA+LR ~84–88%+

# --- Force MLflow to local file store (no server / no proxy) ---
for k in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACTS_URI", "MLFLOW_ENABLE_ARTIFACTS_PROXY"]:
    os.environ.pop(k, None)
mlflow.set_tracking_uri(pathlib.Path("mlruns").resolve().as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)
# ---------------------------------------------------------------

def pick_best(cands):
    return max(cands, key=lambda d: d["val_accuracy"])

def main():
    # 1) อ่านรายชื่อผู้สมัคร (candidates) และเลือกตัวที่ดีที่สุด
    with open(CAND_PATH, "r", encoding="utf-8") as f:
        cands = json.load(f)
    best = pick_best(cands)

    # 2) โหลด test set (ระบุ dtype ลดหน่วยความจำ)
    pixel_dtypes = {f"px{i}": "float32" for i in range(784)}
    dtypes = {**pixel_dtypes, "label": "int8"}
    test = pd.read_csv(SPLITS / "test.csv", dtype=dtypes)
    Xte, yte = test.drop(columns=["label"]).values, test["label"].values

    # 3) โหลดโมเดลจาก run ที่ชนะ แล้วประเมินบน test
    with mlflow.start_run(run_name="4_evaluate_and_save_fashion"):
        model_uri = f"runs:/{best['run_id']}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)
        f1  = f1_score(yte, ypred, average="macro")

        mlflow.log_metrics({"test_accuracy": float(acc), "test_f1_macro": float(f1)})
        mlflow.log_text(classification_report(yte, ypred), "reports/test_classification_report.txt")

        # 4) เซฟโมเดลสำหรับเสิร์ฟ (โฟลเดอร์)
        if DEPLOY_DIR.exists():
            shutil.rmtree(DEPLOY_DIR)
        DEPLOY_DIR.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(model, path=str(DEPLOY_DIR))

    # 5) บันทึกสรุปผลของ run ที่ดีที่สุด + metric บน test
    best.update({"test_accuracy": float(acc), "test_f1_macro": float(f1)})
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    flag = "PASS ✅" if acc >= THRESH else "WARN ⚠"
    print(f"[OK] Evaluation done. test_acc={acc:.4f} ({flag}), model → {DEPLOY_DIR}")

if __name__ == "__main__":
    main()
