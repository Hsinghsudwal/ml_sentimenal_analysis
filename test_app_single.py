import requests


URL = "http://localhost:9696/predict/single"

payload = {
    "text": "This product is amazing and works really well!"
}

response = requests.post(URL, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
