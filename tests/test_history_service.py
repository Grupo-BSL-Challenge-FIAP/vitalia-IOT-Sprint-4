import pytest
from unittest.mock import MagicMock
from api.services.history_service import HistoryService

@pytest.fixture
def mock_repository():
    return MagicMock()

@pytest.fixture
def history_service(mock_repository):
    return HistoryService(repository=mock_repository)

def test_obter_ultimo_registro_existente(history_service, mock_repository):
    mock_repository.obter_informacoes_pet.return_value = {
        "pet_id": 1,
        "birth_date": "2023-01-01",
        "weight_kg": 12.5
    }
    mock_repository.obter_historico_medicoes.return_value = [{
        "measurement_date": "2026-10-01",
        "activity_level": 60.0,
        "sleep_hours": 10.0,
        "heart_rate_bpm": 95.0,
        "respiratory_rate": 22.0,
        "temperature_c": 38.2,
        "behavior_score": 9.0
    }]

    resultado = history_service.obter_ultimo_registro(1)

    assert resultado is not None
    assert resultado["petId"] == 1
    assert resultado["pesoKg"] == 12.5
    assert resultado["atividadePct"] == 60.0

def test_obter_ultimo_registro_inexistente(history_service, mock_repository):
    mock_repository.obter_informacoes_pet.return_value = None
    mock_repository.obter_historico_medicoes.return_value = []

    resultado = history_service.obter_ultimo_registro(9999)
    assert resultado is None

def test_historico_vazio(history_service, mock_repository):
    mock_repository.obter_informacoes_pet.return_value = {"birth_date": "2023-01-01", "weight_kg": 10.0}
    mock_repository.obter_historico_medicoes.return_value = []

    resultado = history_service.obter_ultimo_registro(1)
    assert resultado is None

def test_calcular_medias_historicas(history_service, mock_repository):
    mock_repository.obter_historico_medicoes.return_value = [
        {"activity_level": 70.0, "sleep_hours": 11.0, "temperature_c": 38.5},
        {"activity_level": 50.0, "sleep_hours": 13.0, "temperature_c": 38.1}
    ]

    medias = history_service.calcular_medias_historicas(1)
    
    assert "mediaAtividadePct" in medias
    assert "mediaSonoHoras" in medias
    assert "mediaTemperaturaC" in medias
    assert medias["mediaAtividadePct"] == 50.0

def test_prever_regressao_peso(history_service, mock_repository):
    mock_repository.obter_historico_medicoes.return_value = [
        {"weight_kg": 13.0},
        {"weight_kg": 12.0}
    ]

    regressao = history_service.prever_regressao_peso(1)

    assert regressao is not None
    assert regressao["peso_atual"] == 13.0
    assert regressao["peso_anterior"] == 12.0
    assert regressao["variacao"] == 1.0
    assert regressao["tendencia_peso"] == "SUBIDA"