import requests

URL = "http://localhost:8080/predict/batch"

payload = {
    "texts": [
        "I love this product!",
        "This is terrible and broke immediately."
        "I am okay with old shoes"
    ]
}

response = requests.post(URL, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())

