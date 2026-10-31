import joblib
import os
import pandas as pd
from api.services.history_service import HistoryService

PIPELINE_PATH = "models/trained/vitalia_pipeline.pkl"

class AnalysisService:
    def __init__(self):
        self.history_service = HistoryService()
        if os.path.exists(PIPELINE_PATH):
            self.pipeline = joblib.load(PIPELINE_PATH)
        else:
            self.pipeline = None

    def analisar_comportamento(self, pet_id: int, payload: dict = None):
        ultimo = self.history_service.obter_ultimo_registro(pet_id)
        medias = self.history_service.calcular_medias_historicas(pet_id)

        if not ultimo and not payload:
            return None
        
        if not ultimo:
            ultimo = {
                "idadeAnos": 3.0,
                "pesoKg": 10.0,
                "atividadePct": 50.0,
                "sonoHoras": 12.0,
                "frequenciaCardiacaBpm": 100.0,
                "frequenciaRespiratoria": 24.0,
                "temperaturaC": 38.5,
                "scoreComportamento": 8.0
            }

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
                "mediaPesoKg": ultimo.get("pesoKg", 10.0),
                "mediaAtividadePct": ultimo.get("atividadePct", 50.0),
                "mediaSonoHoras": ultimo.get("sonoHoras", 12.0),
                "mediaTemperaturaC": ultimo.get("temperaturaC", 38.5)
            }
            
        metricas_map = [
            ("peso", float(ultimo.get("pesoKg", 0)), float(medias.get("mediaPesoKg", 0))),
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
            "idade_anos": float(ultimo.get("idadeAnos", 3.0)), 
            "peso_kg": float(ultimo.get("pesoKg", 10.0)),
            "atividade_diaria_pct": float(ultimo.get("atividadePct", 50.0)),
            "sono_horas": float(ultimo.get("sonoHoras", 12.0)),
            "frequencia_cardiaca_bpm": float(ultimo.get("frequenciaCardiacaBpm", 100.0)),
            "frequencia_respiratoria": float(ultimo.get("frequenciaRespiratoria", 24.0)),
            "temperatura_c": float(ultimo.get("temperaturaC", 38.5)),
            "score_comportamento": float(ultimo.get("scoreComportamento", 8.0))
        }])

        status_ml = "NORMAL"
        confidence_val = 1.0
        if self.pipeline:
            pred = self.pipeline.predict(df_input)
            val = str(pred[0])
            mapping = {"0": "NORMAL", "1": "ATTENTION", "2": "RISK"}
            status_ml = mapping.get(val, val)
            
            try:
                probabilidades = self.pipeline.predict_proba(df_input)
                confidence_val = float(max(probabilidades[0]))
            except Exception:
                confidence_val = 1.0

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