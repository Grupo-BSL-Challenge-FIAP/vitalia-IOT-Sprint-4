import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app
from api.security import verificar_autenticacao
from api.exceptions import DatabaseConnectionError

client = TestClient(app)

app.dependency_overrides[verificar_autenticacao] = lambda: "mock_user"

@patch("api.controllers.pet_controller.pipeline")
def test_predict_payload_valido_200(mock_pipeline):
    mock_pipeline.predict.return_value = [0]
    mock_pipeline.predict_proba.return_value = [[0.8, 0.1, 0.1]]

    payload = {
        "idade_anos": 3.0,
        "peso_kg": 12.5,
        "atividade_diaria_pct": 60.0,
        "sono_horas": 12.0,
        "frequencia_cardiaca_bpm": 95.0,
        "frequencia_respiratoria": 22.0,
        "temperatura_c": 38.2,
        "score_comportamento": 9.0
    }

    response = client.post("/api/ai/pets/1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NORMAL"
    assert data["confidence"] == 0.8

def test_predict_payload_invalido_422():
    payload = {
        "idade_anos": "tres_anos",
        "peso_kg": 12.5,
        "atividade_diaria_pct": 60.0,
        "sono_horas": 12.0,
        "frequencia_cardiaca_bpm": 95.0,
        "frequencia_respiratoria": 22.0,
        "temperatura_c": 38.2,
        "score_comportamento": 9.0
    }

    response = client.post("/api/ai/pets/1/predict", json=payload)
    assert response.status_code == 422

@patch("api.services.history_service.HistoryService.obter_ultimo_registro", return_value=None)
def test_pet_inexistente(mock_history):
    response = client.get("/api/ai/pets/9999/dashboard")
    assert response.status_code == 404
    assert response.json()["detail"] == "Pet não encontrado no histórico."

@patch("api.services.analysis_service.AnalysisService.analisar_comportamento")
def test_cenarios_status_ml(mock_analisar):
    mock_analisar.return_value = {
        "status": "NORMAL", "mediaHistorica": 60, "atual": 60, "variacaoPct": 0, "justificativaNumerica": "Estável"
    }
    response = client.get("/api/ai/pets/1/dashboard")
    assert response.status_code == 200
    assert response.json()["status_geral"] == "NORMAL"

    mock_analisar.return_value = {
        "status": "ATTENTION", "mediaHistorica": 60, "atual": 40, "variacaoPct": -33.3, "justificativaNumerica": "Baixa atividade"
    }
    response = client.get("/api/ai/pets/1/dashboard")
    assert response.status_code == 200
    assert response.json()["status_geral"] == "ATTENTION"

    mock_analisar.return_value = {
        "status": "RISK", "mediaHistorica": 60, "atual": 10, "variacaoPct": -83.3, "justificativaNumerica": "Alerta crítico"
    }
    response = client.get("/api/ai/pets/1/dashboard")
    assert response.status_code == 200
    assert response.json()["status_geral"] == "RISK"

@patch("api.controllers.pet_controller.pipeline")
def test_confidence_real(mock_pipeline):
    mock_pipeline.predict.return_value = [2]
    mock_pipeline.predict_proba.return_value = [[0.05, 0.05, 0.90]]

    payload = {
        "idade_anos": 5.0, "peso_kg": 20.0, "atividade_diaria_pct": 10.0,
        "sono_horas": 6.0, "frequencia_cardiaca_bpm": 140.0,
        "frequencia_respiratoria": 30.0, "temperatura_c": 39.5, "score_comportamento": 3.0
    }

    response = client.post("/api/ai/pets/1/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["confidence"] == 0.90

def test_banco_indisponivel():
    with patch("api.services.history_service.HistoryService.obter_ultimo_registro", side_effect=DatabaseConnectionError("Banco de dados temporariamente indisponível.")):
        response = client.get("/api/ai/pets/1/dashboard")
        assert response.status_code == 503
        assert response.json()["detail"] == "Banco de dados temporariamente indisponível."

@patch("api.services.history_service.HistoryService.obter_ultimo_registro", return_value={"idadeAnos": 3.0, "pesoKg": 12.5})
@patch("api.main.AnalysisService.analisar_comportamento")
def test_analyze_payload_valido_200(mock_analisar, mock_history):
    mock_analisar.return_value = {
        "petId": 1,
        "status": "NORMAL",
        "confidence": 0.90,
        "summary": "Análise simulada",
        "recommendations": ["Recomendação"]
    }

    payload = {
        "petId": 1,
        "activityLevel": 60.0,
        "sleepHours": 12.0,
        "heartRate": 95.0,
        "respiratoryRate": 22.0,
        "temperature": 38.2,
        "behaviorScore": 9.0
    }

    response = client.post("/api/ai/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["NORMAL", "ATTENTION", "RISK"]
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0

def test_analyze_payload_invalido_422():
    payload = {
        "petId": "invalido"
    }
    response = client.post("/api/ai/analyze", json=payload)
    assert response.status_code == 422

@patch("api.services.history_service.HistoryService.obter_ultimo_registro", return_value={"idadeAnos": 3.0, "pesoKg": 12.5})
@patch("api.main.AnalysisService.analisar_comportamento")
@pytest.mark.parametrize("status_retorno, probas, esperado", [
    ("NORMAL", [[0.85, 0.10, 0.05]], "NORMAL"),
    ("ATTENTION", [[0.10, 0.80, 0.10]], "ATTENTION"),
    ("RISK", [[0.05, 0.05, 0.90]], "RISK")
])
def test_analyze_cenarios_status(mock_analisar, mock_history, status_retorno, probas, esperado):
    mock_analisar.return_value = {
        "petId": 1,
        "status": esperado,
        "confidence": probas[0][2] if esperado == "RISK" else (probas[0][1] if esperado == "ATTENTION" else probas[0][0]),
        "summary": "Análise simulada",
        "recommendations": ["Recomendação de teste"]
    }

    payload = {
        "petId": 1,
        "activityLevel": 60.0,
        "sleepHours": 12.0,
        "heartRate": 95.0,
        "respiratoryRate": 22.0,
        "temperature": 38.2,
        "behaviorScore": 9.0
    }

    response = client.post("/api/ai/analyze", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == esperado