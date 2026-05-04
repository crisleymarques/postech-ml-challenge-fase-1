from fastapi.testclient import TestClient
from src.main import app
import pytest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_read_health(client):
    """Teste 1: Verifica se a API está online e respondendo."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "operante" in response.json()["message"]


def test_predict_success(client):
    """Teste 2: Verifica se o endpoint de predição funciona com dados corretos."""
    payload = {
        "Gender": "Female",
        "Age": 65,
        "Under30": "No",
        "SeniorCitizen": "Yes",
        "Married": "No",
        "Dependents": "No",
        "NumberofDependents": 0,
        "Country": "United States",
        "State": "California",
        "City": "Los Angeles",
        "ZipCode": 90001,
        "LatLong": "33.973616, -118.242766",
        "Latitude": 33.973616,
        "Longitude": -118.242766,
        "Population": 68701,
        "Quarter_x": "Q3",
        "Quarter_y": "Q3",
        "ReferredaFriend": "No",
        "NumberofReferrals": 0,
        "TenureinMonths": 1,
        "Offer": "None",
        "PhoneService": "Yes",
        "AvgMonthlyLongDistanceCharges": 0.0,
        "MultipleLines": "No",
        "InternetService": "Yes",
        "InternetType": "Fiber Optic",
        "AvgMonthlyGBDownload": 15.0,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtectionPlan": "No",
        "PremiumTechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "StreamingMusic": "No",
        "UnlimitedData": "Yes",
        "Contract": "Month-to-Month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank Withdrawal",
        "MonthlyCharge": 70.0,
        "TotalCharges": 70.0,
        "TotalRefunds": 0.0,
        "TotalExtraDataCharges": 0.0,
        "TotalLongDistanceCharges": 0.0,
        "TotalRevenue": 70.0,
        "SatisfactionScore": 3,
        "CLTV": 5000.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert data["churn_prediction"] in [0, 1]


def test_predict_validation_error(client):
    """Teste 3: Verifica se o Pydantic bloqueia payloads incompletos ou errados."""

    payload = {
        "Gender": "Female",
        "Age": "sessenta_e_cinco",  # Tipo incorreto (espera int)
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
