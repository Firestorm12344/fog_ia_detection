from fastapi import FastAPI, HTTPException
from inference import run_inference
from datetime import datetime

app = FastAPI()

@app.get("/")
def root():
    return {"status": "FUNCIONA"}

@app.post("/api/v1/predict")
def predict(payload: dict):
    try:
        signals = payload["signals"]

        # Validar tamaño ventana
        n = len(signals["ankle"]["x"])
        if n != 128:
            raise ValueError(f"Se esperaban 128 muestras, se recibieron {n}")

        # Ejecutar IA
        result = run_inference(signals)

        # Devolver todo para frontend
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": result["class"],
            "probability": result["probability"],
            "signals": signals
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
