# tests/test_api_unit.py
from fastapi.testclient import TestClient
import tempfile, shutil, pathlib
import mlflow.pyfunc
import numpy as np

# นำแอปจริงมาใช้ แต่จะชี้ DEPLOY_DIR ไปที่โมเดลปลอม
import app.main as api


class _DummyModel(mlflow.pyfunc.PythonModel):
    """โมเดลปลอม: ทายเป็นศูนย์ทั้งหมด"""
    def predict(self, context, model_input):
        return np.zeros((len(model_input),), dtype=int)


def setup_module(module):
    # สร้างโฟลเดอร์ชั่วคราว + เซฟโมเดลปลอมแบบ MLflow pyfunc
    tmp = tempfile.mkdtemp()
    mlflow.pyfunc.save_model(path=tmp, python_model=_DummyModel())
    # ชี้แอปให้โหลดโมเดลจากที่นี่ตอน lifespan startup
    api.DEPLOY_DIR = pathlib.Path(tmp)
    module._TMPDIR = tmp


def teardown_module(module):
    shutil.rmtree(module._TMPDIR, ignore_errors=True)


def test_health_and_predict():
    # ใช้ TestClient → จะเรียก lifespan แล้วโหลดโมเดลปลอมอัตโนมัติ
    with TestClient(api.app) as client:
        # /health
        r = client.get("/health")
        assert r.status_code == 200
        assert "model_path" in r.json()

        # /predict (อินพุต 784 ค่า)
        pixels = [0.0] * 784
        r2 = client.post("/predict", json={"pixels": pixels})
        assert r2.status_code == 200
        body = r2.json()
        assert "label" in body and isinstance(body["label"], int)
        # โมเดลปลอมต้องทาย 0
        assert body["label"] == 0
