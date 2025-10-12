# scripts/1_data_validation.py
# Fashion-MNIST data validation + anomaly detection
import os
import argparse
import pathlib
import json
import numpy as np
import pandas as pd
import mlflow
from sklearn.datasets import fetch_openml

# ---------------- MLflow: force local file store (no server / no proxy) ----------------
for k in ["MLFLOW_TRACKING_URI", "MLFLOW_ARTIFACTS_URI", "MLFLOW_ENABLE_ARTIFACTS_PROXY"]:
    os.environ.pop(k, None)
mlflow.set_tracking_uri(pathlib.Path("mlruns").resolve().as_uri())
mlflow.set_experiment("fashion_mnist_coursework")
# --------------------------------------------------------------------------------------

SEED = 42
ARTIFACTS = pathlib.Path("artifacts")
CSV_PATH = ARTIFACTS / "fashion_mnist.csv"


def _make_tiny_dataframe(n_rows: int = 300) -> pd.DataFrame:
    """สร้างชุดข้อมูลเล็ก (784 พิกเซล 0..255 + label 0..9) สำหรับ CI/ทดสอบเร็ว"""
    rng = np.random.default_rng(SEED)
    X = rng.integers(0, 256, size=(n_rows, 784), dtype=np.uint16)
    y = rng.integers(0, 10, size=(n_rows,), dtype=np.int8)
    df = pd.DataFrame(X, columns=[f"px{i}" for i in range(784)])
    df["label"] = y.astype(int)
    return df


def _load_fashion_mnist() -> pd.DataFrame:
    """ดาวน์โหลด Fashion-MNIST ผ่าน OpenML (ครั้งแรก ~70MB)"""
    fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = pd.DataFrame(fm.data, columns=[f"px{i}" for i in range(fm.data.shape[1])])
    y = pd.Series(fm.target.astype(int), name="label")
    return pd.concat([X, y], axis=1)


def _compute_anomalies(df: pd.DataFrame, sample_n: int = 1000) -> dict:
    """คำนวณ anomalies แบบเบา ๆ"""
    feat = df.drop(columns=["label"])
    # out-of-range (ควรอยู่ 0..255 ใน CSV ดิบ)
    bad_low = int((feat < 0).sum().sum())
    bad_high = int((feat > 255).sum().sum())

    # duplicates
    dups = int(df.duplicated().sum())

    # class imbalance <5%
    counts = df["label"].value_counts()
    total = len(df)
    minority_flags = {str(int(k)): bool(v / total < 0.05) for k, v in counts.items()}

    # z-score outliers (ตัวอย่าง 1k แถว เพื่อความเร็ว)
    sample_n = min(sample_n, len(df))
    sample = feat.sample(n=sample_n, random_state=SEED).astype("float32")
    mean = sample.mean()
    std = sample.std().replace(0, 1)
    z = (sample - mean) / std
    outlier_cells = int((z.abs() > 6).sum().sum())

    return {
        "pixel_out_of_range_low": bad_low,
        "pixel_out_of_range_high": bad_high,
        "duplicate_rows": dups,
        "minority_class_below_5pct": minority_flags,
        "zscore_outlier_cells_gt6_in_sample1k": outlier_cells,
    }


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST validation with anomaly detection")
    parser.add_argument("--tiny", action="store_true",
                        help="ใช้ชุดข้อมูลสุ่มขนาดเล็ก (สำหรับ CI/ทดสอบ เร็ว ไม่โหลดเน็ต)")
    parser.add_argument("--use-existing", action="store_true",
                        help="ถ้ามี artifacts/fashion_mnist.csv อยู่แล้ว ให้ใช้เลย")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # 1) เตรียม DataFrame
    if args.tiny:
        df = _make_tiny_dataframe(n_rows=300)
    elif args.use_existing and CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = _load_fashion_mnist()
        # บันทึกสำเนาดิบไว้ใช้ซ้ำ
        df.to_csv(CSV_PATH, index=False)

    with mlflow.start_run(run_name="1_data_validation_fashion"):
        # 2) บันทึกสถิติหลัก
        mlflow.log_param("dataset", "fashion_mnist" + ("_tiny" if args.tiny else ""))
        mlflow.log_metric("num_rows", int(df.shape[0]))
        mlflow.log_metric("num_cols", int(df.shape[1]))
        mlflow.log_metric("missing_values", int(df.isna().sum().sum()))
        mlflow.log_param("num_classes", int(df["label"].nunique()))

        # ถ้าคนรันสั่ง --use-existing และไฟล์มีอยู่ ให้ log ไฟล์นั้น; มิฉะนั้น log ที่เพิ่งสร้างด้านบน
        if CSV_PATH.exists():
            mlflow.log_artifact(str(CSV_PATH))

        # 3) รายงาน anomaly + checks
        report = {
            "dataset": "fashion_mnist" + ("_tiny" if args.tiny else ""),
            "shape": list(map(int, df.shape)),
            "nulls": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
            "label_unique": sorted(list(map(int, pd.Series(df["label"]).unique()))),
            "basic_checks": {
                "all_numeric": bool(pd.api.types.is_numeric_dtype(df.drop(columns=["label"]).dtypes.unique())),
                "labels_in_0_9": bool(set(pd.Series(df["label"]).unique()).issubset(set(range(10)))),
            },
        }

        anomalies = _compute_anomalies(df, sample_n=1000)
        report["anomalies"] = anomalies

        # สถิติอย่างเร็วของ 1k แถวแรก (เพื่อไม่ให้ไฟล์โตมาก)
        desc_head = df.drop(columns=["label"]).head(1000).describe().to_dict()
        report["desc_numeric_head"] = desc_head

        # 4) log รายงาน
        mlflow.log_dict(report, "reports/validation.json")
        mlflow.log_metrics({
            "anomaly_out_of_range": int(anomalies["pixel_out_of_range_low"] + anomalies["pixel_out_of_range_high"]),
            "anomaly_duplicates": int(anomalies["duplicate_rows"]),
            "anomaly_outliers_cells": int(anomalies["zscore_outlier_cells_gt6_in_sample1k"]),
        })

        # 5) Sanity checks (เฉพาะโหมดปกติ)
        if not args.tiny:
            assert df.shape[0] >= 60000, "Dataset rows too few (<60k)"
            assert df["label"].nunique() >= 5, "Too few classes (<5)"

    print(f"[OK] Validation complete. Data rows={df.shape[0]}, cols={df.shape[1]}")
    if CSV_PATH.exists():
        print("[OK] CSV saved/cached at:", CSV_PATH)


if __name__ == "__main__":
    main()
