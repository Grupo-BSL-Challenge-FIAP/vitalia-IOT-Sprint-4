import joblib
import os
import pandas as pd
import numpy as np
from api.services.history_service import HistoryService
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

PIPELINE_PATH = "models/trained/vitalia_pipeline.pkl"

class AnalysisService:
    def __init__(self):
        self.history_service = HistoryService()
        if os.path.exists(PIPELINE_PATH):
            self.pipeline = joblib.load(PIPELINE_PATH)
            logger.info("Pipeline de machine learning carregado com sucesso.")
        else:
            self.pipeline = None
            logger.warning("Aviso: Pipeline de machine learning não encontrado no caminho especificado.")

    def analisar_comportamento(self, pet_id: int, payload: dict = None):
        logger.info(f"A iniciar análise de comportamento para o petId: {pet_id}")
        
        if not self.pipeline:
            logger.error("Pipeline de machine learning não está disponível.")
            raise HTTPException(status_code=503, detail="Serviço de IA temporariamente indisponível.")

        ultimo = self.history_service.obter_ultimo_registro(pet_id)

        if not ultimo:
            logger.warning(f"Nenhum registo histórico encontrado para o petId: {pet_id}")
            return None

        medias = self.history_service.calcular_medias_historicas(pet_id)

        if payload:
            if "activityLevel" in payload:
                ultimo["atividadePct"] = float(payload["activityLevel"])
            if "sleepHours" in payload:
                ultimo["sonoHoras"] = float(payload["sleepHours"])
            if "heartRate" in payload:
                ultimo["frequenciaCardiacaBpm"] = float(payload["heartRate"])
            if "respiratoryRate" in payload:
                ultimo["frequenciaRespiratoria"] = float(payload["respiratoryRate"])
            if "temperature" in payload:
                ultimo["temperaturaC"] = float(payload["temperature"])
            if "behaviorScore" in payload:
                ultimo["scoreComportamento"] = float(payload["behaviorScore"])

        if not medias:
            medias = {
                "mediaPesoKg": float(ultimo.get("pesoKg", 0.0)),
                "mediaAtividadePct": float(ultimo.get("atividadePct", 0.0)),
                "mediaSonoHoras": float(ultimo.get("sonoHoras", 0.0)),
                "mediaTemperaturaC": float(ultimo.get("temperaturaC", 38.5))
            }
            
        metricas_map = [
            ("atividade", float(ultimo.get("atividadePct", 0)), float(medias.get("mediaAtividadePct", 0))),
            ("sono", float(ultimo.get("sonoHoras", 0)), float(medias.get("mediaSonoHoras", 0))),
            ("temperatura", float(ultimo.get("temperaturaC", 38.5)), float(medias.get("mediaTemperaturaC", 38.5)))
        ]

        alteracoes = []
        for nome, atual, media_hist in metricas_map:
            var_pct = 0.0
            if media_hist > 0:
                var_pct = round(((atual - media_hist) / media_hist) * 100, 2)
            alteracoes.append({
                "metrica": nome,
                "mediaHistorica": media_hist,
                "atual": atual,
                "variacaoPct": var_pct
            })

        df_input = pd.DataFrame([{
            "idade_anos": float(ultimo["idadeAnos"]), 
            "peso_kg": float(ultimo["pesoKg"]),
            "atividade_diaria_pct": float(ultimo.get("atividadePct", 50.0)),
            "sono_horas": float(ultimo.get("sonoHoras", 12.0)),
            "frequencia_cardiaca_bpm": float(ultimo.get("frequenciaCardiacaBpm", 100.0)),
            "frequencia_respiratoria": float(ultimo.get("frequenciaRespiratoria", 24.0)),
            "temperatura_c": float(ultimo.get("temperaturaC", 38.5)),
            "score_comportamento": float(ultimo.get("scoreComportamento", 8.0))
        }])

        try:
            pred = self.pipeline.predict(df_input)
            val = str(pred[0])
            mapping = {"0": "NORMAL", "1": "ATTENTION", "2": "RISK", "NORMAL": "NORMAL", "ATTENTION": "ATTENTION", "RISK": "RISK"}
            
            if val not in mapping:
                logger.error(f"Classe desconhecida retornada pelo modelo no analisar_comportamento: {val}")
                raise HTTPException(status_code=503, detail="Classe inválida retornada pelo modelo de IA.")
                
            status_ml = mapping[val]
            
            probabilidades = self.pipeline.predict_proba(df_input)
            confidence_val = float(max(probabilidades[0]))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erro ao executar a predição do modelo para o petId {pet_id}: {str(e)}")
            raise HTTPException(status_code=503, detail="Erro durante a execução do modelo de IA.")

        media_atividade = float(medias.get("mediaAtividadePct", 0))
        ativ_atual = float(ultimo.get("atividadePct", 0))
        var_ativ = 0.0
        if media_atividade > 0:
            var_ativ = round(((ativ_atual - media_atividade) / media_atividade) * 100, 2)

        tendencia_qualitativa = "CRESCENTE" if var_ativ > 5.0 else ("DECRESCENTE" if var_ativ < -5.0 else "ESTÁVEL")

        linhas_justificativa = []
        for alt in alteracoes:
            nome_metrica = alt["metrica"].capitalize()
            linhas_justificativa.append(
                f"{nome_metrica}: média = {alt['mediaHistorica']}, atual = {alt['atual']}, variação = {alt['variacaoPct']}%"
            )
        
        justificativa_detalhada = " | ".join(linhas_justificativa)

        logger.info(f"Análise concluída com sucesso para o petId: {pet_id} | Status: {status_ml}")

        return {
            "petId": pet_id,
            "mediaHistorica": media_atividade,
            "atual": ativ_atual,
            "variacaoPct": var_ativ,
            "tendencia": tendencia_qualitativa,
            "status": status_ml,
            "confidence": round(confidence_val, 2),
            "alteracoes": alteracoes,
            "justificativaNumerica": justificativa_detalhada
        }

    def prever(self, dados_pet):
        if not self.pipeline:
            logger.error("Pipeline de machine learning não está disponível.")
            raise HTTPException(
                status_code=503, 
                detail="Serviço de IA temporariamente indisponível."
            )
            
        df_entrada = pd.DataFrame([{
            "idade_anos": dados_pet.idade_anos,
            "peso_kg": dados_pet.peso_kg,
            "atividade_diaria_pct": dados_pet.atividade_diaria_pct,
            "sono_horas": dados_pet.sono_horas,
            "frequencia_cardiaca_bpm": dados_pet.frequencia_cardiaca_bpm,
            "frequencia_respiratoria": dados_pet.frequencia_respiratoria,
            "temperatura_c": dados_pet.temperatura_c,
            "score_comportamento": dados_pet.score_comportamento
        }])
        
        try:
            predicao = self.pipeline.predict(df_entrada)
            raw_pred = predicao[0]
        except Exception as e:
            logger.error(f"Erro ao executar predict() no modelo de IA: {str(e)}")
            raise HTTPException(
                status_code=503, 
                detail="Erro durante a execução do modelo de IA."
            )
        
        text_to_id = {"NORMAL": 0, "ATTENTION": 1, "RISK": 2}
        id_to_text = {0: "NORMAL", 1: "ATTENTION", 2: "RISK"}
        
        if isinstance(raw_pred, (int, np.integer)):
            class_id = int(raw_pred)
            if class_id not in id_to_text:
                logger.error(f"Classe numérica desconhecida retornada pelo modelo: {class_id}")
                raise HTTPException(
                    status_code=503, 
                    detail="Classe inválida retornada pelo modelo de IA."
                )
            status_text = id_to_text[class_id]
        else:
            status_text = str(raw_pred).upper()
            if status_text not in text_to_id:
                logger.error(f"Classe textual desconhecida retornada pelo modelo: {status_text}")
                raise HTTPException(
                    status_code=503, 
                    detail="Classe inválida retornada pelo modelo de IA."
                )
            class_id = text_to_id[status_text]
        try:
            probabilidades = self.pipeline.predict_proba(df_entrada)
            confidence = float(np.max(probabilidades[0]))
        except Exception as e:
            logger.error(f"Erro ao calcular confidence do modelo: {str(e)}")
            raise HTTPException(
                status_code=503, 
                detail="Erro durante a execução do modelo de IA."
            )
            
        return class_id, status_text, confidence