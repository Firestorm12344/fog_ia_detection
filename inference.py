import tensorflow as tf
from preprocessing import preprocess
from custom_layers import TimeSeriesAugment


# ==========================================
# REGISTRO CAPAS CUSTOM
# ==========================================
custom_objects = {
    "TimeSeriesAugment": TimeSeriesAugment
}


# ==========================================
# CARGA MODELO
# ==========================================
model = tf.keras.models.load_model(
    "model/modelo.h5",
    custom_objects=custom_objects
)


# ==========================================
# INFERENCIA
# ==========================================
def run_inference(signals: dict):

    # Preprocesamiento
    X = preprocess(signals)  # (1, 128, 9)

    # Predicción
    y = model.predict(X, verbose=0)

    prob = float(y[0][0])

    return {
        "probability": prob,
        "class": "FOG" if prob > 0.5 else "No FOG"
    }
