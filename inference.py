import tensorflow as tf
import numpy as np
from custom_layers import TimeSeriesAugment


# ==========================================
# REGISTRO CAPAS CUSTOM
# ==========================================
custom_objects = {
    "TimeSeriesAugment": TimeSeriesAugment
}


# ==========================================
# CARGA MODELO (UNA SOLA VEZ)
# ==========================================
model = tf.keras.models.load_model(
    "model/modelo.h5",
    custom_objects=custom_objects
)


# ==========================================
# INFERENCIA DIRECTA MATRIZ
# ==========================================
def run_inference(X):

    # Asegurar numpy float32
    X = np.array(X, dtype=np.float32)

    # Asegurar shape (1,128,9)
    if X.ndim == 2:
        X = X[np.newaxis, :, :]

    y = model.predict(X, verbose=0)

    prob = float(y[0][0])

    return {
        "probability": prob,
        "class": "FOG" if prob > 0.5 else "No FOG"
    }
