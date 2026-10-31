import pandas as pd
from datetime import datetime
from api.repositories.pet_repository import PetRepository

class HistoryService:
    def __init__(self, repository: PetRepository = None):
        self.repository = repository or PetRepository()

    def _calcular_idade(self, data_nascimento_str):
        if not data_nascimento_str:
            raise ValueError("Data de nascimento (BIRTH_DATE) não encontrada para o cálculo da idade real.")
        try:
            data_nasc = datetime.strptime(str(data_nascimento_str)[:10], "%Y-%m-%d")
            hoje = datetime.now()
            idade = hoje.year - data_nasc.year - ((hoje.month, hoje.day) < (data_nasc.month, data_nasc.day))
            return max(0, idade)
        except Exception as e:
            raise ValueError(f"Erro ao processar a data de nascimento: {e}")

    def obter_ultimo_registro(self, pet_id: int):
        pet_info = self.repository.obter_informacoes_pet(pet_id)
        medicoes = self.repository.obter_historico_medicoes(pet_id)
        
        if not medicoes:
            return None
            
        ultimo = medicoes[0]
        
        birth_date = pet_info.get("birth_date") if pet_info else None
        idade = self._calcular_idade(birth_date)

        peso_kg = pet_info.get("weight_kg") if pet_info else None
        if peso_kg is None:
            raise ValueError(f"Peso real (WEIGHT_KG) não encontrado para o pet ID {pet_id}.")

        return {
            "petId": int(pet_id),
            "data": str(ultimo.get("measurement_date")),
            "idadeAnos": float(idade),
            "pesoKg": float(peso_kg),
            "atividadePct": float(ultimo.get("activity_level", 50.0)),
            "sonoHoras": float(ultimo.get("sleep_hours", 12.0)),
            "frequenciaCardiacaBpm": float(ultimo.get("heart_rate_bpm", 100.0)),
            "frequenciaRespiratoria": float(ultimo.get("respiratory_rate", 24.0)),
            "temperaturaC": float(ultimo.get("temperature_c", 38.5)),
            "scoreComportamento": float(ultimo.get("behavior_score", 8.0))
        }

    def calcular_medias_historicas(self, pet_id: int):
        medicoes = self.repository.obter_historico_medicoes(pet_id)
        if not medicoes:
            return {}

        df = pd.DataFrame(medicoes)
        historical_df = df.iloc[1:] if len(df) > 1 else df

        return {
            "mediaAtividadePct": float(historical_df['activity_level'].mean().round(1)) if 'activity_level' in historical_df else 0.0,
            "mediaSonoHoras": float(historical_df['sleep_hours'].mean().round(1)) if 'sleep_hours' in historical_df else 12.0,
            "mediaTemperaturaC": float(historical_df['temperature_c'].mean().round(1)) if 'temperature_c' in historical_df else 38.5
        }

    def prever_regressao_peso(self, pet_id: int):
        medicoes = self.repository.obter_historico_medicoes(pet_id)
        if not medicoes or len(medicoes) < 2:
            return None

        pesos = [m.get("weight_kg") for m in medicoes if m.get("weight_kg") is not None]
        if not pesos:
            return None

        peso_atual = pesos[0]
        peso_anterior = pesos[1] if len(pesos) > 1 else peso_atual
        variacao = round(peso_atual - peso_anterior, 2)

        return {
            "peso_atual": float(peso_atual),
            "peso_anterior": float(peso_anterior),
            "variacao": float(variacao),
            "tendencia_peso": "ESTAVEL" if variacao == 0 else ("SUBIDA" if variacao > 0 else "DESCIDA")
        }