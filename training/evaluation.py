import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

PIPELINE_PATH = "models/trained/vitalia_pipeline.pkl"
DATA_PATH = "data/processed/pets_dataset_ready.csv"
METRICS_PATH = "models/metrics.json"

if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"Pipeline não encontrado em {PIPELINE_PATH}.")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset não encontrado em {DATA_PATH}.")

pipeline = joblib.load(PIPELINE_PATH)

df = pd.read_csv(DATA_PATH)
features = [
    "idade_anos", 
    "peso_kg", 
    "atividade_diaria_pct", 
    "sono_horas",
    "frequencia_cardiaca_bpm", 
    "frequencia_respiratoria", 
    "temperatura_c", 
    "score_comportamento"
]
target = "status"

X = df[features]
y = df[target]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = pipeline.predict(X_test)

labels_ordem = ["NORMAL", "ATTENTION", "RISK"]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_ordem).tolist()
}

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print(f"Métricas multiclasse salvas com sucesso em {METRICS_PATH}!")

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred, labels=labels_ordem)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_ordem, 
            yticklabels=labels_ordem)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão Multiclasse")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

print("Matriz de confusão salva com sucesso em models/confusion_matrix.png!")