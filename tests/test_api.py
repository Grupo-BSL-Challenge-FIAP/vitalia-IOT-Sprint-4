import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app
from api.security import verificar_autenticacao

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
    # Testar cenário NORMAL
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
    with patch("api.services.history_service.HistoryService.obter_ultimo_registro", side_effect=Exception("Erro de conexão com o banco Oracle")):
        with pytest.raises(Exception, match="Erro de conexão com o banco Oracle"):
            client.get("/api/ai/pets/1/dashboard")