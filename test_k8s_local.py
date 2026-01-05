import requests

URL = "http://localhost:8000/predict/batch"  # local test with test_k8s.py

payload = {
    "texts": [
        "I absolutely loved this",
        "Terrible experience, waste of money",
        "It was okay, not great"
    ]
}

response = requests.post(URL, json=payload)

print("Status:", response.status_code)
print("Response:")
print(response.json())

