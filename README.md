# 👗 Fashion-MNIST MLOps — MLflow + FastAPI

> เวิร์กโฟลว์ครบวงจร: **Data Validation → Preprocess → Train (หลายรอบ) → Evaluate & Export → Serve API**  
> ✅ ตรงตามที่อาจารย์กำหนด: Public dataset ≥ 10k, ใช้ MLflow, มี Deployment, และมี Tests อย่างน้อย 3 เคส

---

## ✨ Highlights

- **Public Dataset:** Fashion-MNIST (**70,000** แถว, 10 classes) — โหลดผ่าน `sklearn.fetch_openml` ครั้งแรกและแคช
- **Tracking:** MLflow แบบ **local file store** (`./mlruns`) — ไม่ต้องเปิด MLflow server
- **Modeling:** PCA → Logistic Regression (ลองหลายค่า n_components, C) + เก็บ metrics/params
- **Validation:** รายงาน + **Anomaly detection** (OOR pixels, duplicates, imbalance, z-score outliers)
- **Deployment:** FastAPI + Uvicorn พร้อม `/health`, `/predict`, `/predict_batch`
- **Tests:** ≥ 3 เคส (preprocessing, validation anomalies, API via TestClient)
- **CI (GitHub Actions):** รัน step 1 (**tiny mode**) + step 2 + pytest ให้ผ่านเสถียร

---

## 🗺️ Workflow Overview

```mermaid
flowchart LR
    A[1) Data Validation
(Fashion-MNIST, Anomaly Report)] -->|CSV (70k)| B[2) Preprocessing
Scale 0..1 + Train/Val/Test]
    B --> C[3) Training (multi-run)
PCA → LogisticRegression]
    C --> D[4) Evaluation & Export
Pick best + save model]
    D --> E[(FastAPI Serving)]
    A -. MLflow tracking .-> M[(mlruns: params/metrics/artifacts)]
    B -. MLflow tracking .-> M
    C -. MLflow tracking .-> M
    D -. MLflow tracking .-> M
```

---

## 🧰 Environment

แนะนำให้ใช้ Conda (ไฟล์ `environment.yml` มีให้แล้ว)

```bash
# สร้างและเปิดใช้ (Windows / macOS / Linux)
conda env create -f environment.yml
conda activate mlflow-fashion
python -m pip install --upgrade pip
```

> เคลียร์ตัวแปร MLflow ที่อาจเคยตั้งไว้:
```bash
# Windows (PowerShell/CMD)
set MLFLOW_TRACKING_URI=
set MLFLOW_ARTIFACTS_URI=
set MLFLOW_ENABLE_ARTIFACTS_PROXY=

# macOS/Linux (bash/zsh)
unset MLFLOW_TRACKING_URI MLFLOW_ARTIFACTS_URI MLFLOW_ENABLE_ARTIFACTS_PROXY
```

---

## 🚀 Quickstart (รันครบตั้งแต่ต้นจนเสิร์ฟ)

> รันทั้งหมดจากรากโปรเจกต์

```bash
# 1) Data Validation (โหลด 70k แถวครั้งแรก + บันทึก CSV + anomaly report)
python scripts/1_data_validation.py

# 2) Preprocessing (สเกล 0..1 และแบ่ง train/val/test)
python scripts/2_data_preprocessing.py

# 3) Training (หลายรอบ: PCA + LogisticRegression)
python scripts/3_model_training.py

# 4) Evaluation & Export (เลือก best + ประเมิน test + เซฟสำหรับเสิร์ฟ)
python scripts/4_model_evaluation.py
```

เปิด MLflow UI (ตัวเลือก):
```bash
mlflow ui --host 0.0.0.0 --port 5000
# เปิด http://127.0.0.1:5000
```

เสิร์ฟ API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

ทดสอบ:
```bash
# Health
curl http://127.0.0.1:8080/health

# Predict: ใช้สคริปต์ตัวอย่าง
python scripts/test_api.py
```

> 🔎 API จะ **auto-scale** อินพุตที่อยู่ในช่วง 0..255 → 0..1 เพื่อกัน **training–serving skew**

---

## 🔍 Data Validation & Anomaly Detection

เมื่อรัน step 1 แล้ว จะได้:
- `artifacts/fashion_mnist.csv` (70k แถว)
- `mlruns/**/artifacts/reports/validation.json` — ไฟล์รายงานที่มี:
  - ขนาดข้อมูล, missing, unique labels
  - **Anomalies:** out-of-range pixels, duplicates, class imbalance (<5%), z-score outliers (ตัวอย่าง 1k แถว)
  - สถิติเบื้องต้นของคุณสมบัติ (describe)

> โหมดเร็ว (เช่น CI/ออฟไลน์) ใช้ชุดเล็กสุ่ม:
```bash
python scripts/1_data_validation.py --tiny
```

---

## 🧪 Tests

ชุดทดสอบหลัก (≥ 3 เคสตามสเปค):

- `tests/test_preprocessing.py` — ตรวจไฟล์ splits, เช็คสเกล 0..1, โครงสร้างคอลัมน์ 784 + label  
- `tests/test_validation_anomalies.py` — ตรวจ `validation.json` และคีย์ anomalies ที่จำเป็น  
- `tests/test_api.py` — ทดสอบ API ผ่าน **FastAPI TestClient** (ใช้โมเดลปลอม ไม่ต้องเปิด Uvicorn)

รันท้องถิ่น:
```bash
python -m pytest -q --maxfail=1 --disable-warnings
```

> ต้องการ **E2E บนเซิร์ฟเวอร์จริง** ก็มีไฟล์ตัวเลือก `tests/test_api_e2e.py` (มาร์ค `@pytest.mark.e2e`)  
> ใช้: เปิด Uvicorn ก่อน แล้วรัน `python -m pytest -q -m e2e`

---

## 🤖 CI (GitHub Actions)

ไฟล์: `.github/workflows/ci.yml`

- ติดตั้ง deps + flake8 + pytest
- รัน **step 1 (tiny)** เพื่อสร้าง validation.json โดยไม่ดาวน์โหลด 70k จริง
- รัน **step 2** เพื่อสร้าง splits (สเกล 0..1)
- รัน `pytest` (ไม่ต้องเปิดเซิร์ฟเวอร์)

> ถ้าต้องการ E2E จริงใน CI สามารถเพิ่มบล็อกสตาร์ต Uvicorn + ยิง `/health` และ `/predict` ภายหลังได้

---

## 📦 Artifacts & Structure

**Artifacts ที่ถูกสร้าง:**
- `artifacts/fashion_mnist.csv` — ดิบ 70k (หรือ tiny สำหรับ CI)
- `artifacts/splits/{train,val,test}.csv` — ชุดแบ่งสเกล 0..1
- `artifacts/candidates.json` — สรุปผลการเทรนหลายรอบ
- `artifacts/best_run.json` — run ที่ดีที่สุด + metrics test
- `artifacts/deploy/fashion_mnist_clf/` — โมเดลพร้อมเสิร์ฟ

**โครงสร้างโปรเจกต์:**
```
.
├─ scripts/
│  ├─ 1_data_validation.py
│  ├─ 2_data_preprocessing.py
│  ├─ 3_model_training.py
│  └─ 4_model_evaluation.py
├─ app/
│  └─ main.py                 # FastAPI: /health, /predict, /predict_batch
├─ tests/
│  ├─ test_api.py             # TestClient (CI-friendly)
│  ├─ test_preprocessing.py
│  └─ test_validation_anomalies.py
├─ artifacts/                 # (generated)
├─ mlruns/                    # (generated by MLflow)
├─ environment.yml
├─ requirements.txt
└─ README.md
```

**.gitignore (แนะนำ):**
```
mlruns/
artifacts/
data/
datasets/
models/
checkpoints/
*.csv *.npy *.npz *.pkl *.joblib
__pycache__/ *.pyc .DS_Store .vscode/ .idea/
```

---

## 🧑‍🏫 Mapping กับสเปคอาจารย์

- ✅ **Public dataset ≥ 10k** → Fashion-MNIST 70k  
- ✅ **4 ขั้นตอนชัดเจน**: Validation → Preprocess → Train → Evaluate & Export  
- ✅ **MLflow Tracking** + รายงาน anomalies  
- ✅ **Deployment** ด้วย FastAPI + โฟลเดอร์โมเดลพร้อมเสิร์ฟ  
- ✅ **Tests ≥ 3 เคส** (รวม API)  
- ✅ **CI** ที่รันอัตโนมัติและผ่านแน่นอน (แบบไม่โหลดชุดใหญ่)

---

## 🛠️ Troubleshooting

- **“Malformed experiment 'artifacts'”**  
  ลบโฟลเดอร์ `mlruns/artifacts` ที่เผลอสร้างเอง หรือเคลียร์ทั้ง `mlruns/` แล้วรันใหม่
- **MLflow ไปชี้ http://127.0.0.1:5000**  
  ล้างตัวแปร env: `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACTS_URI`, `MLFLOW_ENABLE_ARTIFACTS_PROXY`
- **PCA Warning (“X has feature names…”)**  
  ข้ามได้ หรือส่ง `df.values` ตอน predict ใน API

---

## 📝 License / Credits

- Dataset: **Fashion-MNIST** via `sklearn.fetch_openml("Fashion-MNIST", version=1)`  
- Tracking: **MLflow** (local file store)  
- Serving: **FastAPI + Uvicorn**
