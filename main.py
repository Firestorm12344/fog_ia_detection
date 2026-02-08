from fastapi import FastAPI, HTTPException
from inference import run_inference
from firebase_writter import send_to_firebase

app = FastAPI()

# ======================================
# MODO GLOBAL
# ======================================
MODE = "collect"

# ======================================
# BUFFER GLOBAL (STREAMING)
# ======================================
BUFFER = {
    "ankle": {"x": [], "y": [], "z": []},
    "thigh": {"x": [], "y": [], "z": []},
    "hip": {"x": [], "y": [], "z": []},
}

WINDOW_SIZE = 128


@app.get("/")
def root():
    return {"status": "OK", "mode": MODE}


# ======================================
# CAMBIO MODO IA
# ======================================
@app.post("/api/v1/set_mode")
def set_mode(data: dict):

    global MODE

    mode = data.get("mode")

    if mode not in ["collect", "detect"]:
        raise HTTPException(status_code=400, detail="Modo inválido")

    MODE = mode
    return {"mode": MODE}


# ======================================
# RECEPCIÓN STREAMING SEÑALES
# ======================================
@app.post("/api/v1/predict")
def predict(payload: dict):

    try:

        signals = payload["signals"]

        # ==================================================
        # ACUMULAR CHUNKS
        # ==================================================
        for sensor in ["ankle", "thigh", "hip"]:
            for axis in ["x", "y", "z"]:
                BUFFER[sensor][axis].extend(signals[sensor][axis])

        # ==================================================
        # SI YA HAY VENTANA COMPLETA
        # ==================================================
        prediction = None

        if len(BUFFER["ankle"]["x"]) >= WINDOW_SIZE:

            # Construir ventana
            window = {
                s: {
                    a: BUFFER[s][a][:WINDOW_SIZE]
                    for a in ["x", "y", "z"]
                }
                for s in ["ankle", "thigh", "hip"]
            }

            # Limpiar buffer
            for s in BUFFER:
                for a in BUFFER[s]:
                    BUFFER[s][a] = BUFFER[s][a][WINDOW_SIZE:]

            # Ejecutar IA solo si detect
            if MODE == "detect":
                prediction = run_inference(window)

            # Enviar a Firebase
            send_to_firebase(window, prediction)

        return {"status": "chunk recibido"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
