from fastapi import FastAPI, HTTPException
from inference import run_inference

app = FastAPI()

# ==========================================
# MODO GLOBAL
# ==========================================
current_mode = "collect"  # collect | detect


# ==========================================
# ROOT TEST
# ==========================================
@app.get("/")
def root():
    return {
        "status": "Servidor activo",
        "mode": current_mode
    }


# ==========================================
# TRIGGER CAMBIO DE MODO
# ==========================================
@app.post("/api/v1/set_mode")
def set_mode(payload: dict):

    global current_mode

    mode = payload.get("mode")

    if mode not in ["collect", "detect"]:
        raise HTTPException(status_code=400, detail="Modo inválido")

    current_mode = mode

    return {
        "status": "modo actualizado",
        "mode": current_mode
    }


# ==========================================
# RECEPCIÓN DE SEÑALES
# ==========================================
@app.post("/api/v1/predict")
def predict(payload: dict):

    try:
        signals = payload["signals"]

        # Validación básica
        n = len(signals["ankle"]["x"])
        if n != 128:
            raise ValueError(
                f"Se esperaban 128 muestras, se recibieron {n}"
            )

        # ---------------------------
        # SOLO RECOLECCIÓN
        # ---------------------------
        if current_mode == "collect":

            return {
                "mode": current_mode,
                "signals": signals,
                "prediction": None
            }

        # ---------------------------
        # DETECCIÓN IA
        # ---------------------------
        prediction = run_inference(signals)

        return {
            "mode": current_mode,
            "signals": signals,
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
