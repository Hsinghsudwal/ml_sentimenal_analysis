import requests
import time

URL = "http://localhost:30001/predict/single"

# payload = {"text": "Great product!"}
payload = {
    "texts": [
        "Great product!",
        "I love this!",
        "This is terrible",
        "It's okay, not great"
    ]
}

while True:
    response = requests.post(URL, json=payload)
    print(response.json())
    time.sleep(0.1)
