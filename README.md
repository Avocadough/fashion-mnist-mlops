
# ML Project — MLflow Workflow (Iris)

โครงสร้างตรงตามที่กำหนด พร้อมสคริปต์ 4 ขั้นตอน:
1) Data Validation → 2) Data Preprocessing → 3) Model Training (หลายรอบ) → 4) Model Evaluation & Save for Serving (FastAPI)

## Quickstart
```bash
# 0) ติดตั้งไลบรารี
pip install -r requirements.txt

# 1) ตั้งค่า (ถ้าไม่เปิด MLflow UI ก็ยังรันได้)
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}
export MLFLOW_EXPERIMENT=iris_coursework
export ACCURACY_THRESHOLD=0.90

# 2) รันตามลำดับงาน
python scripts/1_data_validation.py
python scripts/2_data_preprocessing.py
python scripts/3_model_training.py
python scripts/4_model_evaluation.py

# 3) เปิด API (ต้องมีโมเดลถูกบันทึกจากขั้นตอน 4)
python app/main.py
# ทดสอบ
curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json"   -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Artifacts ที่ถูกสร้าง:
- `artifacts/iris.csv` — ข้อมูลทั้งหมด
- `artifacts/splits/{train,val,test}.csv` — ชุดแบ่งข้อมูล
- `artifacts/candidates.json` — ผลลัพธ์รันฝึกหลายรอบ
- `artifacts/best_run.json` — สรุป run ที่ดีที่สุด (อ้างอิงโดยสคริปต์ประเมิน)
- `artifacts/deploy/iris_clf/` — โฟลเดอร์โมเดลสำหรับเสิร์ฟด้วย FastAPI

## โครงสร้าง
```
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
```
