import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import insurance_api.main as main

    importlib.reload(main)  # pick up MODEL_PATH
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_has_metrics(client):
    info = client.get("/model-info").json()
    assert {"f1", "roc_auc", "pr_auc"} <= info["metrics"].keys()
    assert 0 < info["threshold"] < 1


def test_predict_returns_probability_and_reasons(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["interested"] == (body["probability"] >= body["threshold"])
    assert len(body["top_reasons"]) == 5
    features = [reason["feature"] for reason in body["top_reasons"]]
    assert len(features) == len(set(features)), "one-hot columns should be merged per feature"


def test_insured_customer_scores_lower(client, customer):
    not_insured = client.post("/predict", json=customer).json()["probability"]
    insured = client.post(
        "/predict", json={**customer, "Previously_Insured": 1, "Vehicle_Damage": "No"}
    ).json()["probability"]
    assert insured < not_insured


def test_unseen_region_is_handled(client, customer):
    r = client.post("/predict", json={**customer, "Region_Code": 59})
    assert r.status_code == 200


def test_batch(client, customer):
    r = client.post("/predict/batch", json={"customers": [customer, {**customer, "Age": 25}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "bad",
    [{"Age": 10}, {"Gender": "X"}, {"Vehicle_Age": "3 Years"}, {"Annual_Premium": -5}],
)
def test_invalid_input_rejected(client, customer, bad):
    assert client.post("/predict", json={**customer, **bad}).status_code == 422


def test_no_model_gives_503(monkeypatch, tmp_path, customer):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    import insurance_api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=customer).status_code == 503
