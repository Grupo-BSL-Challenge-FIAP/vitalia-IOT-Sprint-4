import pytest
from unittest.mock import patch, MagicMock
from api.services.llm_service import LLMService

@pytest.fixture
def llm_service_mocked():
    """Fixture que inicializa o LLMService."""
    return LLMService()

def test_fallback_do_gemini_sem_cliente():
    """Testa o fallback quando a API Key está ausente."""
    with patch("api.services.llm_service.os.getenv", return_value=None):
        service = LLMService()
        resultado = service.answer("Contexto do pet", "Pergunta?")
        assert "[Modo Fallback - IA Indisponível/Sem Chave]" in resultado

@patch("api.services.llm_service.requests.post")
def test_funcionamento_normal_do_gemini(mock_post, llm_service_mocked):
    """Testa o funcionamento bem-sucedido da API REST do Gemini."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Resposta gerada pela IA com sucesso."}]
            }
        }]
    }
    mock_post.return_value = mock_response

    resultado = llm_service_mocked.answer({"nome": "Rex"}, "O que ele tem?")
    assert "Resposta gerada pela IA" in resultado

@patch("api.services.llm_service.requests.post")
def test_fallback_do_gemini_por_erro_servidor(mock_post, llm_service_mocked):
    """Testa o fallback quando a API do Gemini retorna erro de servidor."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"
    mock_post.return_value = mock_response

    resultado = llm_service_mocked.answer("Contexto de Risco", "O que se passa?")
    assert "[Modo Fallback de Segurança]" in resultado

@pytest.mark.parametrize("status_ml", ["NORMAL", "ATTENTION", "RISK"])
def test_cenarios_pet(status_ml, llm_service_mocked):
    """Testa diferentes cenários de status do pet."""
    contexto = {"nome": "Thor", "status": status_ml}
    with patch("api.services.llm_service.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": f"Análise para status {status_ml}"}]
                }
            }]
        }
        mock_post.return_value = mock_response
        
        resultado = llm_service_mocked.answer(contexto, "Como ele está?")
        assert f"Análise para status {status_ml}" in resultado