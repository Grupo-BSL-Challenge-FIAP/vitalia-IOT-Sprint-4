from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

class MeasurementInput(BaseModel):
    petId: int
    activityLevel: float = Field(...)
    sleepHours: float = Field(...)
    heartRate: float = Field(...)
    respiratoryRate: float = Field(...)
    temperature: float = Field(...)
    behaviorScore: float = Field(...)

class PetDataInput(BaseModel):
    idade_anos: float = Field(ge=0, le=40)
    peso_kg: float = Field(gt=0)
    atividade_diaria_pct: float = Field(ge=0, le=100)
    sono_horas: float = Field(ge=0, le=24)
    frequencia_cardiaca_bpm: float = Field(ge=30, le=250)
    frequencia_respiratoria: float = Field(ge=5, le=100)
    temperatura_c: float = Field(ge=30, le=45)
    score_comportamento: float = Field(ge=1, le=10)
    historico_medicoes: Optional[List[float]] = Field(default=None)

    @field_validator(
        'idade_anos', 'peso_kg', 'atividade_diaria_pct', 'sono_horas',
        'frequencia_cardiaca_bpm', 'frequencia_respiratoria', 'temperatura_c', 'score_comportamento',
        mode='before'
    )
    def validar_tipos_invalidos(cls, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Tipo inválido. O valor deve ser um número numérico (inteiro ou flutuante).")
        return float(value)

    @model_validator(mode='after')
    def validar_historico(self):
        if self.historico_medicoes is not None and len(self.historico_medicoes) == 0:
            raise ValueError("O histórico de medições não pode estar vazio.")
        if self.historico_medicoes is not None and len(self.historico_medicoes) > 100:
            raise ValueError("Tamanho incompatível. O histórico excede o limite máximo de 100 registros.")
        return self

class QuestionInput(BaseModel):
    pergunta: str = Field(..., min_length=1, description="A pergunta não pode estar vazia.")

class InsightOutput(BaseModel):
    pet_id: str
    status: str
    mensagem_alerta: str
    explicabilidade: dict 
    insight_ia: str
    
class RecommendationInput(BaseModel):
    dados_atuais: PetDataInput