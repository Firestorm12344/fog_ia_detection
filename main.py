from fastapi import FastAPI, HTTPException
from inference import run_inference

app = FastAPI()

@app.get("/")
def root():
    return {"status": "FUNCIONA"}

@app.post("/api/v1/predict")
def predict(payload: dict):
    try:
        signals = payload["signals"]

        n = len(signals["ankle"]["x"])
        if n != 128:
            raise ValueError(f"Se esperaban 128 muestras, se recibieron {n}")

        return run_inference(signals)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
