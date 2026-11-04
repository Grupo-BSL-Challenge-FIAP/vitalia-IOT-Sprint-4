import os
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")

    def answer(self, contexto, pergunta: str) -> str:
        if not self.api_key:
            return "[Modo Fallback - IA Indisponível/Sem Chave]"

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

        prompt = f"""
            Você é o assistente virtual inteligente da Vitalia AI, especializado em saúde e comportamento animal.
            Responda de forma completa, natural, inteligente e detalhada à pergunta do tutor, utilizando o contexto do pet abaixo quando relevante, e respondendo livremente a qualquer dúvida apresentada (como alimentação, cuidados, etc.).

            Contexto estruturado do pet:
            {contexto_formatado}

            Pergunta do tutor:
            {pergunta}
        """

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-3.5-flash-lite:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=body, timeout=15)
            if response.status_code == 200:
                data = response.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return f"Resposta gerada para: {contexto_formatado} - {pergunta}"
            else:
                return f"[Modo Fallback de Segurança]"
        except Exception as e:
            return f"[Modo Fallback de Segurança]"

llm_service = LLMService()