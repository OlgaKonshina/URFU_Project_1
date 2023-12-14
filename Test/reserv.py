from fastapi.testclient import TestClient
from fastapi import FastAPI
client = TestClient(FastAPI())

# predict_positive():
response = client.post("https://huggingface.co/spaces/OlgaKo/emotional_coloring_of_the_text", json = {'text':'Мне очень нравится учиться!'})
json_data = response.json()

print(json_data)
