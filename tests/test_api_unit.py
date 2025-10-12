# tests/test_api_unit.py
from fastapi.testclient import TestClient
import tempfile, shutil, pathlib
import mlflow.pyfunc
import numpy as np

# นำแอปจริงมาใช้ แล้วชี้ DEPLOY_DIR ไปยังโมเดลปลอมสำหรับเทสต์
import app.main as api


class _DummyModel(mlflow.pyfunc.PythonModel):
    """โมเดลปลอม: ทำนายเป็น 0 เสมอ"""
    def predict(self, context, model_input):
        n = len(model_input)
        return np.zeros((n,), dtype=int)


_TMPDIR = None

def setup_module(module=None):
    global _TMPDIR
    _TMPDIR = tempfile.mkdtemp()
    mlflow.pyfunc.save_model(path=_TMPDIR, python_model=_DummyModel())
    api.DEPLOY_DIR = pathlib.Path(_TMPDIR)

def teardown_module(module=None):
    shutil.rmtree(_TMPDIR, ignore_errors=True)


def test_health_and_predict_single():
    with TestClient(api.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

        pixels = [0.0] * 784
        r2 = client.post("/predict", json={"pixels": pixels})
        assert r2.status_code == 200
        body = r2.json()
        assert "label" in body and isinstance(body["label"], int)


def test_predict_batch():
    with TestClient(api.app) as client:
        batch = [[0.0] * 784 for _ in range(3)]
        r = client.post("/predict_batch", json={"batch": batch})
        assert r.status_code == 200

        body = r.json()
        # API ปัจจุบันคืน {"count": int, "results": [ {label:int, label_name:str}, ... ]}
        assert isinstance(body, dict)
        assert "count" in body and "results" in body
        assert isinstance(body["count"], int) and body["count"] == 3
        assert isinstance(body["results"], list) and len(body["results"]) == 3
        for item in body["results"]:
            assert isinstance(item, dict)
            assert "label" in item and isinstance(item["label"], int)
            # label_name เป็น optional ถ้า API ใส่มาให้ก็เช็คว่าเป็น str
            if "label_name" in item:
                assert isinstance(item["label_name"], str)
