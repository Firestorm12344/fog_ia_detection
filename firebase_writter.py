import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase_admin_key.json")

firebase_admin.initialize_app(cred)

db = firestore.client()


def send_to_firebase(signals, prediction=None):

    data = {
        "signals": signals,
        "prediction": prediction
    }

    db.collection("fog_signals").add(data)
