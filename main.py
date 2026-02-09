from fastapi import FastAPI, HTTPException
import numpy as np
from inference import run_inference
from firebase_writter import send_to_firebase

app = FastAPI()

MODE = "collect"

BUFFER = []
WINDOW = 128


@app.post("/api/v1/predict")
def predict(payload: dict):

    try:

        data = payload["d"]

        BUFFER.extend(data)

        prediction = None

        if len(BUFFER) >= WINDOW:

            window = np.array(BUFFER[:WINDOW])
            BUFFER[:] = BUFFER[WINDOW:]

            if MODE == "detect":
                prediction = run_inference_matrix(window)

            send_to_firebase(window.tolist(), prediction)

        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def run_inference_matrix(X):

    X = np.array(X, dtype=np.float32)
    X = X[np.newaxis, :, :]  # (1,128,9)

    return run_inference(X)
