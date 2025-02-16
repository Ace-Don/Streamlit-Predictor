import requests

# Replace with your API key
API_TOKEN = "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"
MODEL = "scikit-learn/random-forest"
API_URL = "https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

def predict(features):
    payload = {"inputs": [features]}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

features = [5.1, 3.5, 1.4, 0.2]  # Example input
result = predict(features)
print(result)
