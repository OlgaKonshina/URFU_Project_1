from fastapi.testclient import TestClient
from Api import app

client = TestClient(app)


def test_predict():
    response = client.post("/predict/",
        json={"text": "It was a wonderful journey. We have visited many beautiful places and seen many sightings! I am happy!"}
    )
    json_data = response.json() 

    assert response.status_code == 200
    assert json_data['Результат:'] == 'позитивный :)'
    
