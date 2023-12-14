from fastapi.testclient import TestClient
from app_text_emotion import app

client = TestClient(app)

#тест проверяет доступность приложения при обращении к корню сервера
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

#тест пытается распознать тональность положительной фразы
def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "Мне очень нравится учиться!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'Текст позитивный'

#тест пытается распознать тональность негативной фразы
def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "Я устала и хочу домой!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'Текст негативный'

#тест пытается распознать тональность нейтральной фразы
def test_predict_neutral():
        response = client.post("/predict/",
                               json={"text": "Сегодня наступила весна."})
        json_data = response.json()
        assert response.status_code == 200
        assert json_data['label'] == 'Текст нейтральный'
