import tensorflow as tf
from preprocessing import preprocess
from custom_layers import TimeSeriesAugment

custom_objects = {
    "TimeSeriesAugment": TimeSeriesAugment
}

model = tf.keras.models.load_model(
    "model/modelo.h5",
    custom_objects=custom_objects
)


def run_inference(signals: dict):

    X = preprocess(signals)  # (1,128,9)

    y = model.predict(X, verbose=0)

    prob = float(y[0][0])

    return {
        "probability": prob,
        "class": "FOG" if prob > 0.5 else "No FOG"
    }
