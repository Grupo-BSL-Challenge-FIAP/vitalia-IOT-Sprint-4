# Usa uma imagem oficial e leve do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas o arquivo de dependências primeiro (otimiza o build)
COPY requirements.txt .

# Instala as bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto
COPY . .

# Expõe a porta que a API vai utilizar
EXPOSE 8000

# Comando para iniciar o servidor da API
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
