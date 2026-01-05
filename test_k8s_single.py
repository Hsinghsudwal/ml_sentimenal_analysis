import requests

URL = "http://localhost:30001/predict/single"

payload = {
    "text": "This product is amazing and works really well!"
}

response = requests.post(URL, json=payload)

print("Status:", response.status_code)
print("Response:")
print(response.json())
