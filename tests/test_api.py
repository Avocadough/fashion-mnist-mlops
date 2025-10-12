import json, requests, pandas as pd

# อ่านแถวแรกจาก test.csv
df = pd.read_csv("artifacts/splits/test.csv")
row = df.iloc[1]
pixels = row.drop("label").astype(float).tolist()  # 784 ค่า (สเกล 0..1 จากสคริปต์ preprocess)

# ยิงไปที่ API
url = "http://127.0.0.1:8000/predict"
payload = {"pixels": pixels}
res = requests.post(url, json=payload, timeout=30)
print("HTTP", res.status_code, res.reason)
print("true_label:", int(row["label"]))
print("response:", res.json())
