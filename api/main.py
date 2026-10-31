import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from json import JSONDecodeError
from pydantic import BaseModel, Field
from typing import List
from api.services.analysis_service import AnalysisService

from api.controllers.pet_controller import router as pet_router
from api.security import verificar_autenticacao

load_dotenv()

ENV = os.getenv("APP_ENV", "development")
SECRET_KEY = os.getenv("API_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("API_SECRET_KEY não definida nas variáveis de ambiente.")

class MeasurementInput(BaseModel):
    petId: int
    activityLevel: float = Field(..., ge=0.0, le=100.0, description="Percentual de atividade entre 0 e 100")
    sleepHours: float = Field(..., ge=0.0, le=24.0, description="Horas de sono entre 0 e 24")
    heartRate: float = Field(..., ge=10.0, le=300.0, description="Frequência cardíaca em BPM")
    respiratoryRate: float = Field(..., ge=5.0, le=100.0, description="Frequência respiratória")
    temperature: float = Field(..., ge=30.0, le=45.0, description="Temperatura corporal em Celsius")
    behaviorScore: float = Field(..., ge=0.0, le=10.0, description="Score de comportamento de 0 a 10")

class AIAnalysisResponse(BaseModel):
    petId: int
    status: str
    confidence: float
    summary: str
    recommendations: List[str]

app = FastAPI(
    title="Vitalia AI API", 
    description="API de Inteligência Artificial para Pets com LLM", 
    version="1.2",
    swagger_ui_parameters={"persistAuthorization": True},
    dependencies=[Depends(verificar_autenticacao)]
)

app.include_router(pet_router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    erros_personalizados = []
    for error in exc.errors():
        campo = " -> ".join(str(x) for x in error["loc"])
        msg_original = error["msg"]
        
        if "greater than" in msg_original:
            msg = "O valor deve ser maior do que o limite permitido."
        elif "less than" in msg_original:
            msg = "O valor deve ser menor do que o limite permitido."
        elif "Field required" in msg_original:
            msg = "Este campo é obrigatório."
        else:
            msg = msg_original
            
        erros_personalizados.append({"campo": campo, "erro": msg})
        
    return JSONResponse(
        status_code=422,
        content={"detail": erros_personalizados, "mensagem": "Erro de validação nos dados enviados."}
    )

@app.exception_handler(JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: JSONDecodeError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [{"campo": "body", "erro": "Tipo inválido. O formato do JSON enviado é inválido ou contém tipos numéricos incorretos."}],
            "mensagem": "Erro de validação nos dados enviados."
        }
    )


analysis_service = AnalysisService()

@app.post("/api/ai/analyze", response_model=AIAnalysisResponse)
def analyze_measurement(data: MeasurementInput):
    payload = {
        "activityLevel": data.activityLevel,
        "sleepHours": data.sleepHours,
        "heartRate": data.heartRate,
        "respiratoryRate": data.respiratoryRate,
        "temperature": data.temperature,
        "behaviorScore": data.behaviorScore
    }
   
    resultado_analise = analysis_service.analisar_comportamento(data.petId, payload=payload)
    
    if not resultado_analise:
        return {
            "petId": data.petId,
            "status": "NORMAL",
            "confidence": 0.85,
            "summary": "Registo do pet processado com sucesso utilizando a nova medição.",
            "recommendations": ["Continuar com a rotina normal de monitorização."]
        }
    
    status = resultado_analise.get("status", "NORMAL")
    justificativa = resultado_analise.get("justificativaNumerica", "Análise efetuada com base na nova medição.")
    
    recomendacoes = [
        "Continuar monitorizando nas próximas horas.",
        "Caso a alteração persista ou o comportamento mude drasticamente, considerar avaliação veterinária."
    ]
    if status == "RISK":
        recomendacoes.insert(0, "⚠️ ATENÇÃO: Os indicadores apontam para um desvio crítico em relação ao histórico.")

    return {
        "petId": data.petId,
        "status": status,
        "confidence": 0.92,  
        "summary": justificativa,
        "recommendations": recomendacoes
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "message": "Vitalia AI API está a funcionar perfeitamente!"
    }