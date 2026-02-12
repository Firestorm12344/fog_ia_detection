import firebase_admin
from firebase_admin import credentials, firestore
import os, json

firebase_key = json.loads(os.environ["FIREBASE_KEY"])

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def send_to_firebase(signals, prediction=None):

    data = {
        "signals": signals,
        "prediction": prediction
    }
    print("Enviando a Firebase...")
    db.collection("fog_signals").add(data)
