from fastapi.testclient import TestClient
from src.service.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    payload = {"texts": ["I love this hotel", "This is terrible"]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2

def test_home_page_html():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    # basic HTML sanity checks
    assert "<title>Sentiment Analysis</title>" in response.text
    assert "<form action=\"/predict_ui\" method=\"post\">" in response.text
    assert "Analyze" in response.text


def test_predict_ui_html():
    form_data = {"text": "Amazing hotel, I loved it!"}

    response = client.post("/predict_ui", data=form_data)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    # check that the response contains expected elements
    assert "Sentiment Analysis Result" in response.text
    assert "Predicted sentiment:" in response.text
    assert "Analyze another sentence" in response.text

# to run the tests, run on terminal: pytest