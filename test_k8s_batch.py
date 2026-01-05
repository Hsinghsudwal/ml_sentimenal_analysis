import requests

URL = "http://localhost:30001/predict/batch"

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

