นี่คือไฟล์ README.md (ฉบับอัปเดตสำหรับโปรเจกต์ Fashion-MNIST + MLflow + FastAPI) — ก็อปวางทับไฟล์เดิมได้เลย:

# ML Project — MLflow Workflow (Fashion-MNIST)

เวิร์กโฟลว์ครบ 4 ขั้นตอน:  
**1) Data Validation → 2) Data Preprocessing → 3) Model Training (หลายรอบ) → 4) Model Evaluation & Save for Serving (FastAPI)**

- **Public dataset:** Fashion-MNIST (70,000 แถว, 10 คลาส)  
- โหลดผ่าน `sklearn.fetch_openml("Fashion-MNIST", version=1)` ครั้งแรกเท่านั้น แล้วเก็บเป็น `artifacts/fashion_mnist.csv`  
- **MLflow:** ใช้ **local file store** (`./mlruns`) ไม่ต้องเปิด MLflow server

## Quickstart

```bash
# 0) ติดตั้งไลบรารี
pip install -r requirements.txt

# 1) รันตามลำดับงาน
python scripts/1_data_validation.py      # ดึง Fashion-MNIST + ตรวจสอบ + เซฟ CSV
python scripts/2_data_preprocessing.py   # แบ่ง train/val/test และสเกล 0..1
python scripts/3_model_training.py       # เทรนหลายรอบ (PCA+LogReg) + log ที่ MLflow
python scripts/4_model_evaluation.py     # เลือก best → ประเมินบน test → เซฟโมเดลสำหรับเสิร์ฟ

# 2) เปิด API (ต้องมีโมเดลจากขั้นตอน 4 แล้ว)
# ทางเลือก A: uvicorn CLI (รันที่รากโปรเจกต์)
uvicorn app.main:app --host 0.0.0.0 --port 8080

# ทางเลือก B: รันเป็นสคริปต์ (ถ้าไฟล์ app/main.py ใช้ uvicorn.run(app, ...))
# python app/main.py

ทดสอบ API
# ตรวจสุขภาพ
curl http://127.0.0.1:8080/health


ตัวอย่างทดสอบด้วย Python (อ่านหนึ่งภาพจาก test.csv แล้วยิงไป /predict):

# scripts/test_api.py
import requests, pandas as pd
df = pd.read_csv("artifacts/splits/test.csv")
row = df.iloc[0]
pixels = row.drop("label").astype(float).tolist()
r = requests.post("http://127.0.0.1:8080/predict", json={"pixels": pixels}, timeout=30)
print("true_label:", int(row["label"]))
print("response:", r.json())


รัน:

python scripts/test_api.py

Artifacts ที่ถูกสร้าง

artifacts/fashion_mnist.csv — ข้อมูลทั้งหมด (70k แถว)

artifacts/splits/{train,val,test}.csv — ชุดแบ่งข้อมูล (ประมาณ 56k / 7k / 7k)

artifacts/candidates.json — ผลรันการฝึกหลายรอบ (PCA + ค่าพารามิเตอร์ C)

artifacts/best_run.json — สรุป run ที่ดีที่สุด + metric บน test

artifacts/deploy/fashion_mnist_clf/ — โฟลเดอร์โมเดลสำหรับเสิร์ฟด้วย FastAPI

โครงสร้าง
ml-project/
├─ .github/workflows/main.yml
├─ scripts/
│  ├─ 1_data_validation.py
│  ├─ 2_data_preprocessing.py
│  ├─ 3_model_training.py
│  └─ 4_model_evaluation.py
├─ tests/
│  └─ test_preprocessing.py
├─ app/
│  └─ main.py
├─ Dockerfile
├─ requirements.txt
└─ README.md

MLflow (โลคัล)

สคริปต์ตั้งค่า mlflow.set_tracking_uri(file:///.../mlruns) ให้โดยอัตโนมัติ — ถ้าอยากเปิด UI:

mlflow ui --host 0.0.0.0 --port 5000
# เปิดเบราว์เซอร์ที่ http://127.0.0.1:5000

.gitignore ที่แนะนำ (กันไฟล์ใหญ่ขึ้น Git)

ไม่ควร push artifacts/, mlruns/, และไฟล์ dataset ที่สร้างซ้ำได้

ดูตัวอย่างในไฟล์ .gitignore ของโปรเจกต์นี้