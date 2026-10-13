import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    # /predict returns 200 with {"error": ...} on failure, so check the body too
    body = response.json()
    assert "error" not in body
    assert body["prediction"] in {"Likely to churn", "Not likely to churn"}


def test_predict_rejects_missing_field(client, customer):
    del customer["Contract"]
    assert client.post("/predict", json=customer).status_code == 422


def test_predict_rejects_wrong_type(client, customer):
    customer["tenure"] = "twelve"
    assert client.post("/predict", json=customer).status_code == 422


def test_gradio_ui_is_mounted(client):
    assert client.get("/ui/").status_code == 200
