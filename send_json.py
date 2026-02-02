import requests
import numpy as np

URL = "http://127.0.0.1:8000/api/v1/predict"

N = 128  # 👈 número correcto de muestras

payload = {
    "signals": {
        "ankle": {
            "x": np.random.randn(N).tolist(),
            "y": np.random.randn(N).tolist(),
            "z": np.random.randn(N).tolist()
        },
        "thigh": {
            "x": np.random.randn(N).tolist(),
            "y": np.random.randn(N).tolist(),
            "z": np.random.randn(N).tolist()
        },
        "hip": {
            "x": np.random.randn(N).tolist(),
            "y": np.random.randn(N).tolist(),
            "z": np.random.randn(N).tolist()
        }
    }
}

r = requests.post(URL, json=payload)
print(r.json())
