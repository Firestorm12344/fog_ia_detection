from fastapi import FastAPI, HTTPException
from inference import run_inference
import os

app = FastAPI()

# ======================================================
# MODO GLOBAL
# ======================================================
MODE = os.getenv("MODE", "collect")  # collect | detect


# ======================================================
# ROOT TEST
# ======================================================
@app.get("/")
def root():
    return {"status": "FUNCIONA", "mode": MODE}


# ======================================================
# CAMBIAR MODO
# ======================================================
@app.post("/api/v1/set_mode")
def set_mode(data: dict):
    global MODE

    mode = data.get("mode")

    if mode not in ["collect", "detect"]:
        raise HTTPException(400, "Modo inválido")

    MODE = mode

    return {
        "mode": MODE,
        "message": "Modo actualizado"
    }


# ======================================================
# RECEPCIÓN DE SEÑALES
# ======================================================
@app.post("/api/v1/signals")
def receive_signals(payload: dict):

    global MODE

    try:
        signals = payload["signals"]

        # Validación básica
        n = len(signals["ankle"]["x"])
        if n != 128:
            raise ValueError(
                f"Se esperaban 128 muestras, se recibieron {n}"
            )

        # ---------- SOLO RECOLECTAR ----------
        if MODE == "collect":
            return {
                "mode": MODE,
                "signals": signals,
                "status": "recibido"
            }

        # ---------- IA ACTIVADA ----------
        prediction = run_inference(signals)

        return {
            "mode": MODE,
            "prediction": prediction,
            "signals": signals
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
