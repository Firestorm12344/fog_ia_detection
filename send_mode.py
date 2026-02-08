import requests

URL = "https://fog-ia-detection-3.onrender.com/api/v1/set_mode"

response = requests.post(
    URL,
    json={"mode": "collect"}   # 👈 "collect" o "detect"
)

print("Status:", response.status_code)
print("Respuesta:", response.json())
