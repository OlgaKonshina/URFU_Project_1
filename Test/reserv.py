from fastapi.testclient import TestClient
from fastapi import FastAPI
client = TestClient(FastAPI())

# predict_positive():
response = client.post("/predict/", json = {'text':'Мне очень нравится учиться!'})
json_data = response.json()

print(json_data)
