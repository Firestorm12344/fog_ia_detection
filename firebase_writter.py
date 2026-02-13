import firebase_admin
from firebase_admin import credentials, firestore
import os, json

# Inicializar Firebase UNA sola vez
if not firebase_admin._apps:
    firebase_key = json.loads(os.environ["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_key)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def send_to_firebase(data):
    try:
        db.collection("fog_signals").add(data)
        print("Firebase write OK")
    except Exception as e:
        print("Firebase ERROR:", e)
