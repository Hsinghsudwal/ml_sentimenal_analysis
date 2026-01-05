import requests

URL = "http://localhost:8080/predict/single"

payload = {"text": "This product is amazing and works really well!"}

response = requests.post(URL, json=payload)
print(response.status_code)
print(response.json())
