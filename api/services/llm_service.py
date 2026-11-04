import os
import time
from dotenv import load_dotenv
from google import genai

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
            return "[Modo Fallback - Sem Chave de API configurada]"

        prompt = f"""
            Você é o assistente virtual inteligente da Vitalia AI, especializado em saúde e comportamento animal.
            Responda de forma completa, natural, inteligente e detalhada à pergunta do tutor, utilizando o contexto do pet abaixo quando relevante, e respondendo livremente a qualquer dúvida apresentada.

            Contexto estruturado do pet:
            {contexto_formatado}

            Pergunta do tutor:
            {pergunta}
        """

        max_retries = 2
        for tentativa in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-3.5-flash-lite',
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text
                except Exception:
                    pass

                if tentativa == max_retries - 1:
                    return f"[Erro na API do Gemini]: {str(e)}"
                time.sleep(1)

        return "[Erro de Conexão com a IA]"

llm_service = LLMService()