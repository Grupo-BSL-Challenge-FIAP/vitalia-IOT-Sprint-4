import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.exceptions import DatabaseConnectionError
from api.services.analysis_service import AnalysisService
from api.services.history_service import HistoryService
from api.services.llm_service import llm_service
from api.security import verificar_autenticacao
from api.schemas.pet_schemas import PetDataInput, QuestionInput, MeasurementInput

router = APIRouter(prefix="/api/ai/pets", tags=["Pets AI"])
     
STATUS_MAP = {0: "NORMAL", 1: "ATTENTION", 2: "RISK"}

history_service = HistoryService()
analysis_service = AnalysisService()

@router.get("/{pet_id}/insights")
async def get_pet_insights_get(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID do pet inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
    
    status_analise = resultado["status"]
    insights = resultado.get("justificativaNumerica", "Comportamento analisado com base no histórico recente.")
    ultimo = history_service.obter_ultimo_registro(pid_int)
        
    return {
        "pet_id": pet_id, 
        "status_analise": status_analise, 
        "insights": insights,
        "ultimo_registro": ultimo
    }

@router.get("/{pet_id}/dashboard")
async def get_pet_dashboard(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    return {
        "pet_id": pet_id,
        "status_geral": resultado["status"],
        "indicadores_consolidados": {
            "media_atividade_pct": resultado["mediaHistorica"],
            "atividade_atual_pct": resultado["atual"],
            "reducao_atividade_pct": abs(resultado["variacaoPct"]),
            "justificativa": resultado["justificativaNumerica"]
        }
    }

@router.get("/{pet_id}/recommendations")
async def get_pet_recommendations(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    recomendacoes = []
    if resultado["status"] == "RISK":
        recomendacoes.append({
            "categoria": "Acompanhamento", 
            "acao": "Caso as alterações persistam, considerar avaliação veterinária.", 
            "justificativa": resultado["justificativaNumerica"]
        })
    elif resultado["status"] == "ATTENTION":
        recomendacoes.append({
            "categoria": "Acompanhamento", 
            "acao": "Acompanhar as alterações identificadas nos próximos dias.", 
            "justificativa": resultado["justificativaNumerica"]
        })
    else:
        recomendacoes.append({
            "categoria": "Manutenção", 
            "acao": "Acompanhar as alterações identificadas nos próximos dias.", 
            "justificativa": resultado["justificativaNumerica"]
        })
        
    return {
        "pet_id": pet_id,
        "status_analise": resultado["status"],
        "total_recomendacoes": len(recomendacoes),
        "recomendacoes": recomendacoes
    }
    
@router.post("/{pet_id}/predict")
async def prever_comportamento(pet_id: str, dados_pet: PetDataInput, credentials: str = Depends(verificar_autenticacao)):
    class_id, status_text, confidence = analysis_service.prever(dados_pet)
        
    try:
        pid_formatted = int(pet_id)
    except ValueError:
        pid_formatted = pet_id

    return {
        "petId": pid_formatted,
        "classId": class_id,
        "status": status_text,
        "confidence": round(confidence, 2)
    }

@router.post("/{pet_id}/ask")
async def perguntar_ao_pet(pet_id: str, body: QuestionInput, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    pergunta_texto = body.pergunta.strip()
    if not pergunta_texto:
        raise HTTPException(status_code=422, detail="A pergunta não pode estar vazia.")
        
    pet_info = history_service.repository.obter_informacoes_pet(pid_int)
    
    ultimo = history_service.obter_ultimo_registro(pid_int)
    if not ultimo:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    medicoes = history_service.repository.obter_historico_medicoes(pid_int)
    resultado_analise = analysis_service.analisar_comportamento(pid_int)
    
    if not resultado_analise:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")

    contexto = {
        "nome": pet_info.get("nome") or pet_info.get("name", f"Pet {pet_id}") if pet_info else f"Pet {pet_id}",
        "especie": pet_info.get("especie") or pet_info.get("species", "Não informada") if pet_info else "Não informada",
        "idade": ultimo.get("idadeAnos"),
        "peso": ultimo.get("pesoKg"),
        "frequenciaCardiaca": ultimo.get("frequenciaCardiacaBpm"),
        "frequenciaRespiratoria": ultimo.get("frequenciaRespiratoria"),
        "temperatura": ultimo.get("temperaturaC"),
        "behaviorScore": ultimo.get("scoreComportamento"),
        "historicoRecente": medicoes[:5] if medicoes else [], 
        "status": resultado_analise.get("status"),
        "alteracoes": resultado_analise.get("justificativaNumerica", "Nenhuma alteração significativa detetada.")
    }
    
    resposta = llm_service.answer(contexto, pergunta_texto)
    
    return {
        "petId": pid_int,
        "pergunta": pergunta_texto,
        "resposta": resposta
    }

@router.get("/{pet_id}/trends")
async def get_pet_trends(pet_id: str, credentials: str = Depends(verificar_autenticacao)):
    try:
        pid_int = int(pet_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID inválido.")
        
    resultado = analysis_service.analisar_comportamento(pid_int)
    if not resultado:
        raise HTTPException(status_code=404, detail="Pet não encontrado no histórico.")
        
    medicoes = history_service.repository.obter_historico_medicoes(pid_int)
    if not medicoes:
        raise HTTPException(status_code=422, detail="Histórico insuficiente para calcular tendências.")

    df = pd.DataFrame(medicoes)

    return {
        "pet_id": pet_id,
        "tendencia_geral": resultado.get("tendencia"),
        "status_analise": resultado.get("status"),
        "variacao_pct_atividade": resultado.get("variacaoPct"),
        "media_historica_atividade": resultado.get("mediaHistorica"),
        "atividade_atual": resultado.get("atual"),
        "tendencias_reais": {
            "atividade": df['activity_level'].tolist() if 'activity_level' in df else [],
            "sono": df['sleep_hours'].tolist() if 'sleep_hours' in df else [],
            "temperatura": df['temperature_c'].tolist() if 'temperature_c' in df else [],
            "frequencia_cardiaca": df['heart_rate_bpm'].tolist() if 'heart_rate_bpm' in df else [],
            "frequencia_respiratoria": df['respiratory_rate'].tolist() if 'respiratory_rate' in df else [],
            "behavior_score": df['behavior_score'].tolist() if 'behavior_score' in df else []
        },
        "detalhes": resultado.get("justificativaNumerica")
    }

@router.post("/{petId}/measurements/analyze")
def analisar_medicao_pet(petId: int, payload: MeasurementInput, credentials: str = Depends(verificar_autenticacao)):
    if payload.petId != petId:
        raise HTTPException(
            status_code=400,
            detail="O ID do pet na URL não corresponde ao ID do payload."
        )
    
    try:
        resultado = analysis_service.analisar_comportamento(petId, payload.model_dump())
        
        if not resultado:
            raise HTTPException(status_code=404, detail="Pet não encontrado no histórico ou base indisponível.")

        status = resultado["status"] 
        
        if status == "RISK":
            recomendacao = "⚠️ ATENÇÃO: Os indicadores apontam para um desvio crítico. Considerar avaliação veterinária."
        elif status == "ATTENTION":
            recomendacao = "Acompanhe as alterações identificadas nas próximas horas."
        else:
            recomendacao = "Acompanhe as alterações nas próximas horas."

        return {
            "petId": petId,
            "status": status,
            "confidence": resultado.get("confidence", 0.90),
            "recommendation": recomendacao,
            "summary": resultado.get("justificativaNumerica", "Análise realizada com sucesso.")
        }
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail="Serviço de IA temporariamente indisponível.")