import pytest
from unittest.mock import MagicMock, patch
from api.services.analysis_service import AnalysisService

@pytest.fixture
def analysis_service():
    service = AnalysisService()
    service.pipeline = MagicMock()
    return service

def test_analisar_comportamento_cenario_normal(analysis_service):
    analysis_service.history_service.obter_ultimo_registro = MagicMock(return_value={
        "idadeAnos": 3.0, "pesoKg": 10.0, "atividadePct": 60.0, "sonoHoras": 12.0,
        "frequenciaCardiacaBpm": 100.0, "frequenciaRespiratoria": 24.0, "temperaturaC": 38.5, "scoreComportamento": 8.0
    })
    analysis_service.history_service.calcular_medias_historicas = MagicMock(return_value={
        "mediaAtividadePct": 60.0, "mediaSonoHoras": 12.0, "mediaTemperaturaC": 38.5
    })
    
    analysis_service.pipeline.predict.return_value = ["0"]  # "0" mapeia para NORMAL
    analysis_service.pipeline.predict_proba.return_value = [[0.05, 0.05, 0.90]]

    resultado = analysis_service.analisar_comportamento(pet_id=1)
    
    assert resultado is not None
    assert resultado["status"] == "NORMAL"
    assert resultado["confidence"] == 0.9

def test_analisar_comportamento_cenario_attention(analysis_service):
    analysis_service.history_service.obter_ultimo_registro = MagicMock(return_value={
        "idadeAnos": 3.0, "pesoKg": 10.0, "atividadePct": 40.0, "sonoHoras": 10.0,
        "frequenciaCardiacaBpm": 110.0, "frequenciaRespiratoria": 24.0, "temperaturaC": 38.5, "scoreComportamento": 7.0
    })
    analysis_service.history_service.calcular_medias_historicas = MagicMock(return_value={
        "mediaAtividadePct": 60.0, "mediaSonoHoras": 12.0, "mediaTemperaturaC": 38.5
    })
    
    analysis_service.pipeline.predict.return_value = ["1"]  # "1" mapeia para ATTENTION
    analysis_service.pipeline.predict_proba.return_value = [[0.10, 0.82, 0.08]]

    resultado = analysis_service.analisar_comportamento(pet_id=1)
    
    assert resultado is not None
    assert resultado["status"] == "ATTENTION"
    assert resultado["confidence"] == 0.82

def test_analisar_comportamento_cenario_risk(analysis_service):
    analysis_service.history_service.obter_ultimo_registro = MagicMock(return_value={
        "idadeAnos": 3.0, "pesoKg": 10.0, "atividadePct": 10.0, "sonoHoras": 8.0,
        "frequenciaCardiacaBpm": 150.0, "frequenciaRespiratoria": 35.0, "temperaturaC": 39.8, "scoreComportamento": 4.0
    })
    analysis_service.history_service.calcular_medias_historicas = MagicMock(return_value={
        "mediaAtividadePct": 60.0, "mediaSonoHoras": 12.0, "mediaTemperaturaC": 38.5
    })
    
    analysis_service.pipeline.predict.return_value = ["2"]  # "2" mapeia para RISK
    analysis_service.pipeline.predict_proba.return_value = [[0.01, 0.10, 0.89]]

    resultado = analysis_service.analisar_comportamento(pet_id=1)
    
    assert resultado is not None
    assert resultado["status"] == "RISK"
    assert resultado["confidence"] == 0.89

def test_analisar_comportamento_confidence_e_variacoes(analysis_service):
    analysis_service.history_service.obter_ultimo_registro = MagicMock(return_value={
        "idadeAnos": 3.0, "pesoKg": 10.0, "atividadePct": 70.0, "sonoHoras": 12.0,
        "frequenciaCardiacaBpm": 100.0, "frequenciaRespiratoria": 24.0, "temperaturaC": 38.5, "scoreComportamento": 8.0
    })
    analysis_service.history_service.calcular_medias_historicas = MagicMock(return_value={
        "mediaAtividadePct": 60.0, "mediaSonoHoras": 12.0, "mediaTemperaturaC": 38.5
    })
    
    analysis_service.pipeline.predict.return_value = ["0"]
    analysis_service.pipeline.predict_proba.return_value = [[0.02, 0.0, 0.98]]

    resultado = analysis_service.analisar_comportamento(pet_id=1)
    
    assert resultado is not None
    assert "confidence" in resultado
    assert resultado["confidence"] == 0.98
    assert "alteracoes" in resultado
    assert "tendencia" in resultado