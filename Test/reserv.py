from fastapi.testclient import TestClient
from app_text_emotion import app

client = TestClient(app)

#тест пытается распознать тональность положительной фразы
def _predict_positive():
    response = client.post("/predict/",
                           json={"text": "Мне очень нравится учиться!"})
    json_data = response.json()
    response.status_code == 200
    print(json_data)
