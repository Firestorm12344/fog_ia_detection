import tensorflow as tf
from preprocessing import preprocess
from custom_layers import TimeSeriesAugment


# Registrar capa personalizada
custom_objects = {
    "TimeSeriesAugment": TimeSeriesAugment
}

# Cargar modelo
model = tf.keras.models.load_model(
    "model/modelo.h5",
    custom_objects=custom_objects
)


def run_inference(signals: dict):

    # Preprocesar señales
    X = preprocess(signals)   # (1,128,9)

    y = model.predict(X, verbose=0)

    prob = float(y[0][0])

    return {
        "probability": prob,
        "class": "FOG" if prob > 0.5 else "No FOG"
    }
