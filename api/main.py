import os
import logging
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from json import JSONDecodeError
from pydantic import BaseModel, Field
from typing import List
from fastapi import FastAPI, Depends, Request, HTTPException
from api.services.analysis_service import AnalysisService

from api.controllers.pet_controller import router as pet_router, pipeline
from api.security import verificar_autenticacao
from api.exceptions import DatabaseConnectionError
import joblib
import pandas as pd
import numpy as np
from api.controllers import pet_controller

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

ENV = os.getenv("APP_ENV", "development")
SECRET_KEY = os.getenv("API_SECRET_KEY")
if not SECRET_KEY:
    logger.error("Tentativa de inicialização sem API_SECRET_KEY definida.")
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

PIPELINE_PATH = os.path.join(os.path.dirname(__file__), '../models/trained/vitalia_pipeline.pkl')
try:
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("Pipeline de Machine Learning carregado com sucesso.")
except Exception as e:
    logger.warning(f"Não foi possível carregar o pipeline de ML: {e}")
    pipeline = None

analysis_service = AnalysisService()

@app.post("/api/ai/analyze", response_model=AIAnalysisResponse)
def analyze_measurement(data: MeasurementInput):
    status_text = "NORMAL"
    confidence = 0.85

    idade_real = 3.0
    peso_real = 12.5
    try:
        from api.services.history_service import HistoryService
        history_svc = HistoryService()
        ultimo = history_svc.obter_ultimo_registro(data.petId)
        if ultimo:
            idade_real = float(ultimo.get("idadeAnos", 3.0))
            peso_real = float(ultimo.get("pesoKg", 12.5))
    except Exception as e:
        logger.warning(f"Não foi possível obter idade e peso reais do histórico: {e}")

    if pet_controller.pipeline:
        df_entrada = pd.DataFrame([{
            "idade_anos": idade_real, 
            "peso_kg": peso_real,
            "atividade_diaria_pct": data.activityLevel,
            "sono_horas": data.sleepHours,
            "frequencia_cardiaca_bpm": data.heartRate,
            "frequencia_respiratoria": data.respiratoryRate,
            "temperatura_c": data.temperature,
            "score_comportamento": data.behaviorScore
        }])
        try:
            predicao = pet_controller.pipeline.predict(df_entrada)
            raw_pred = predicao[0]
            id_to_text = {0: "NORMAL", 1: "ATTENTION", 2: "RISK"}
            
            if isinstance(raw_pred, (int, np.integer)):
                status_text = id_to_text.get(int(raw_pred), "NORMAL")
            else:
                status_text = str(raw_pred).upper()

            probabilidades = pet_controller.pipeline.predict_proba(df_entrada)
            confidence = float(np.max(probabilidades[0]))
        except Exception as e:
            logger.error(f"Erro durante a predição do modelo: {e}")

    try:
        from api.services.history_service import HistoryService
        history_svc = HistoryService()
        history_svc.obter_ultimo_registro(data.petId)
    except Exception as e:
        if "conexão" in str(e).lower() or "database" in str(e).lower() or "oracle" in str(e).lower():
            logger.error(f"Erro de conexão com o banco de dados: {e}")
            raise HTTPException(status_code=503, detail="Banco de dados temporariamente indisponível.")
        raise e

    justificativa = "Análise efetuada com base na nova medição e histórico recente do pet."
    
    recomendacoes = [
        "Continuar monitorizando nas próximas horas.",
        "Caso a alteração persista ou o comportamento mude drasticamente, considerar avaliação veterinária."
    ]
    if status_text == "RISK":
        recomendacoes.insert(0, "⚠️ ATENÇÃO: Os indicadores apontam para um desvio crítico em relação ao histórico.")

    return {
        "petId": data.petId,
        "status": status_text,
        "confidence": round(confidence, 2), 
        "summary": justificativa,
        "recommendations": recomendacoes
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "message": "Vitalia AI API está a funcionar perfeitamente!"
    }

@app.exception_handler(DatabaseConnectionError)
async def database_connection_exception_handler(request: Request, exc: DatabaseConnectionError):
    logger.error(f"DatabaseConnectionError capturado: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Banco de dados temporariamente indisponível."},
    )