# app/main.py
import pathlib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from contextlib import asynccontextmanager
import uvicorn

DEPLOY_DIR = pathlib.Path("artifacts/deploy/fashion_mnist_clf")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class FashionItem(BaseModel):
    # one image: 784 floats (0..1), แต่ถ้าพลั้งส่ง 0..255 จะสเกลให้อัตโนมัติ
    pixels: conlist(float, min_length=784, max_length=784)

class FashionBatch(BaseModel):
    batch: list[conlist(float, min_length=784, max_length=784)]

# --- Use FastAPI lifespan (แทน on_event) เพื่อตัด DeprecationWarning ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DEPLOY_DIR.exists():
        raise RuntimeError("Deploy dir not found. Run scripts/4_model_evaluation.py first.")
    app.state.model = mlflow.pyfunc.load_model(str(DEPLOY_DIR))
    yield
    # (optional) cleanup here

app = FastAPI(title="Fashion-MNIST API (MLflow)", lifespan=lifespan)
# -------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(DEPLOY_DIR)}

def _to_dataframe(pixels_list: list[float]) -> pd.DataFrame:
    df = pd.DataFrame([pixels_list], columns=[f"px{i}" for i in range(784)])
    # auto-scale: ถ้าเจอค่าสูงกว่า 1.0 ถือว่า 0..255 แล้วสเกลลง 0..1
    if df.to_numpy().max() > 1.0:
        df = df.astype("float32") / 255.0
    return df

@app.post("/predict")
def predict(item: FashionItem):
    try:
        df = _to_dataframe(item.pixels)
        pred = int(app.state.model.predict(df)[0])
        name = CLASS_NAMES[pred] if 0 <= pred < len(CLASS_NAMES) else str(pred)
        return {"label": pred, "label_name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/predict_batch")
def predict_batch(items: FashionBatch):
    try:
        df = pd.DataFrame(items.batch, columns=[f"px{i}" for i in range(784)])
        if df.to_numpy().max() > 1.0:
            df = df.astype("float32") / 255.0
        preds = app.state.model.predict(df)
        results = []
        for p in preds:
            p = int(p)
            results.append({"label": p, "label_name": CLASS_NAMES[p] if 0 <= p < len(CLASS_NAMES) else str(p)})
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")

if __name__ == "__main__":
    # ✅ ส่งอ็อบเจกต์ app โดยตรง เพื่อให้ทำงานได้แม้รันเป็นสคริปต์
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
