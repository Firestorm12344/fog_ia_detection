from fastapi import FastAPI, HTTPException, Request
import numpy as np
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


@app.get("/health")
def health():
    return {"ok": True}


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
# RECEPCIÓN BINARIA STREAMING
# ======================================
@app.post("/api/v1/predict")
async def predict(request: Request):

    try:

        raw = await request.body()

        # Convertir bytes → int16
        data = np.frombuffer(raw, dtype=np.int16)

        # Cada fila = 9 valores
        data = data.reshape(-1, 9)

        print("Samples recibidos:", len(data))

        # Convertir al formato signals original
        signals = {
            "ankle": {"x": [], "y": [], "z": []},
            "thigh": {"x": [], "y": [], "z": []},
            "hip": {"x": [], "y": [], "z": []},
        }

        for row in data:

            signals["ankle"]["x"].append(int(row[0]))
            signals["ankle"]["y"].append(int(row[1]))
            signals["ankle"]["z"].append(int(row[2]))

            signals["thigh"]["x"].append(int(row[3]))
            signals["thigh"]["y"].append(int(row[4]))
            signals["thigh"]["z"].append(int(row[5]))

            signals["hip"]["x"].append(int(row[6]))
            signals["hip"]["y"].append(int(row[7]))
            signals["hip"]["z"].append(int(row[8]))

        # Acumular buffer
        for sensor in ["ankle", "thigh", "hip"]:
            for axis in ["x", "y", "z"]:
                BUFFER[sensor][axis].extend(signals[sensor][axis])

        prediction = None

        if len(BUFFER["ankle"]["x"]) >= WINDOW_SIZE:

            window = {
                s: {
                    a: BUFFER[s][a][:WINDOW_SIZE]
                    for a in ["x", "y", "z"]
                }
                for s in ["ankle", "thigh", "hip"]
            }

            # limpiar buffer
            for s in BUFFER:
                for a in BUFFER[s]:
                    BUFFER[s][a] = BUFFER[s][a][WINDOW_SIZE:]

            if MODE == "detect":
                prediction = run_inference(window)

            send_to_firebase(window, prediction)

        return {"status": "binario recibido"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
