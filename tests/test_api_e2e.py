# tests/test_api_e2e.py
import time
import requests
import pandas as pd
from pathlib import Path
import pytest

API_URL = "http://127.0.0.1:8000"

@pytest.mark.e2e
def test_e2e_health_and_predict():
    # รอให้เซิร์ฟเวอร์พร้อม (เช็ค /health สูงสุด ~10s)
    for _ in range(20):
        try:
            r = requests.get(f"{API_URL}/health", timeout=1.0)
            if r.ok:
                break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("API server is not up at /health")

    # เตรียม payload: ถ้ามี test.csv ใช้แถวจริง, ไม่มีก็ใช้ zeros
    test_csv = Path("artifacts/splits/test.csv")
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        row = df.iloc[0]
        pixels = row.drop("label").astype(float).tolist()
    else:
        pixels = [0.0] * 784  # fallback

    res = requests.post(f"{API_URL}/predict", json={"pixels": pixels}, timeout=5)
    assert res.status_code == 200
    body = res.json()
    assert "label" in body and isinstance(body["label"], int)
