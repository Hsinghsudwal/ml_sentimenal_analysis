import requests

url = "http://localhost:9696/predict/batch"

payload = {
    "texts": [
        "I love this!",
        "This is terrible",
        "It's okay, not great"
    ]
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())