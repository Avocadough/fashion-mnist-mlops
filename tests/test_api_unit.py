# tests/test_api_unit.py
from fastapi.testclient import TestClient
import tempfile, shutil, pathlib
import mlflow.pyfunc
import numpy as np

# ใช้แอปจริง แต่ชี้ DEPLOY_DIR ไปยังโมเดลปลอมใน temp
import app.main as api


class _DummyModel(mlflow.pyfunc.PythonModel):
    """โมเดลปลอม: ทายเป็น 0 ทั้งหมด"""
    def predict(self, context, model_input):
        n = len(model_input)
        return np.zeros((n,), dtype=int)


_TMPDIR = None

def setup_module(module=None):
    global _TMPDIR
    _TMPDIR = tempfile.mkdtemp()
    mlflow.pyfunc.save_model(path=_TMPDIR, python_model=_DummyModel())
    # ให้แอปโหลดโมเดลจาก temp นี้ตอน startup
    api.DEPLOY_DIR = pathlib.Path(_TMPDIR)

def teardown_module(module=None):
    shutil.rmtree(_TMPDIR, ignore_errors=True)


def test_health_and_predict_single():
    with TestClient(api.app) as client:
        # /health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

        # /predict (784 ค่า)
        pixels = [0.0] * 784
        r2 = client.post("/predict", json={"pixels": pixels})
        assert r2.status_code == 200
        assert r2.json().get("label") == 0


def test_predict_batch():
    with TestClient(api.app) as client:
        batch = [[0.0] * 784 for _ in range(3)]
        r = client.post("/predict_batch", json={"batch": batch})
        assert r.status_code == 200
        preds = r.json()
        assert isinstance(preds, list) and len(preds) == 3
