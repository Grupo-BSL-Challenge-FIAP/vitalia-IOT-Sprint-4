import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError, ServerError

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def answer(self, contexto, pergunta: str) -> str:
        if isinstance(contexto, dict):
            nome = contexto.get("nome", "Não informado")
            especie = contexto.get("especie", "Não informada")
            idade = contexto.get("idade", contexto.get("idadeAnos", "Não informada"))
            peso = contexto.get("peso", contexto.get("pesoKg", "Não informado"))
            fc = contexto.get("frequenciaCardiaca", contexto.get("frequenciaCardiacaBpm", "Não informada"))
            fr = contexto.get("frequenciaRespiratoria", "Não informada")
            temperatura = contexto.get("temperatura", contexto.get("temperaturaC", "Não informada"))
            behavior_score = contexto.get("behaviorScore", contexto.get("scoreComportamento", "Não informado"))
            historico = contexto.get("historicoRecente", contexto.get("historico", "Nenhum histórico recente disponível"))
            status_ml = contexto.get("status", "NORMAL")
            alteracoes = contexto.get("alteracoes", "Nenhuma alteração significativa detetada")

            contexto_formatado = f"""
                - Nome do pet: {nome}
                - Espécie: {especie}
                - Idade: {idade} anos
                - Peso: {peso} kg
                - Frequência cardíaca: {fc} bpm
                - Frequência respiratória: {fr} rpm
                - Temperatura: {temperatura} °C
                - Behavior score: {behavior_score}
                - Histórico recente: {historico}
                - Status retornado pelo ML: {status_ml}
                - Alterações detetadas: {alteracoes}
            """
        else:
            contexto_formatado = str(contexto)

        if not self.client:
            return f"[Modo Fallback - IA Indisponível/Sem Chave] Com base nos dados analisados do pet:\n{contexto_formatado}\n\nResposta padrão: Não foi possível contactar o serviço de IA no momento, mas os dados do pet encontram-se registados e seguros no sistema."

        prompt = f"""
Você é o assistente virtual da Vitalia AI. A Vitalia AI é um apoio ao acompanhamento e à decisão.

Diretrizes obrigatórias:
- Não invente informações.
- Não realize diagnóstico.
- Não afirme que o animal possui uma doença.
- Não prescreva medicamentos ou tratamentos.
- Oriente avaliação veterinária quando necessário.
- O status do pet vem estritamente do modelo de Machine Learning (NORMAL, ATTENTION ou RISK). Utilize-o para explicar os resultados sem alterar a classe.

Contexto estruturado do pet:
{contexto_formatado}

Pergunta do tutor:
{pergunta}
"""
        max_retries = 2
        for tentativa in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except (ServerError, APIError) as e:
                if tentativa == max_retries - 1:
                    break
                time.sleep(1)
            except Exception:
                break

        return f"[Modo Fallback de Segurança] O serviço de IA encontra-se temporariamente indisponível. Resumo baseado no contexto do pet:\n{contexto_formatado}\n\nRecomendação: Verifique os parâmetros vitais registados e consulte um médico veterinário em caso de dúvida."

llm_service = LLMService()