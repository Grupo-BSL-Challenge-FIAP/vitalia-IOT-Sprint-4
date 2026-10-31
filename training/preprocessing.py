import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(filepath: str, num_records: int = 2000):
    """Gera dados simulados de comportamento e saúde alinhados ao schema Oracle."""
    np.random.seed(42)
    
    pet_ids = np.random.randint(1001, 1100, num_records)
    for i in range(10):
        pet_ids[i] = 1000
    
    unique_pets = np.unique(pet_ids)
    pet_ages = {pid: round(float(np.random.uniform(0.5, 15)), 1) for pid in unique_pets}
    pet_ages[1000] = 3.0
    idades_fixas = [pet_ages[pid] for pid in pet_ids]
    
    pet_base_weights = {pid: round(float(np.random.uniform(4.0, 25.0)), 2) for pid in unique_pets}
    pet_base_weights[1000] = 12.5  
    pesos_variados = [
        round(float(np.clip(pet_base_weights[pid] + np.random.normal(0, 0.2), 1.0, 50.0)), 2) 
        for pid in pet_ids
    ]
    
    datas_base = []
    pet_counters = {}
    hoje = datetime.now()
    for i, pid in enumerate(pet_ids):
        if pid == 1000:
            data_pet_1000 = hoje - timedelta(days=(15 - i))
            datas_base.append(data_pet_1000.strftime("%Y-%m-%d"))
        else:
            if pid not in pet_counters:
                pet_counters[pid] = hoje - timedelta(days=np.random.randint(10, 30))
            else:
                pet_counters[pid] += timedelta(days=np.random.randint(1, 3))
                if pet_counters[pid] > hoje:
                    pet_counters[pid] = hoje - timedelta(days=np.random.randint(0, 5))
            datas_base.append(pet_counters[pid].strftime("%Y-%m-%d"))

   
    atividade_diaria = [round(float(np.clip(np.random.normal(70, 10), 10, 100)), 2) for _ in range(num_records)]
    sono_horas = [round(float(np.clip(np.random.normal(12, 2), 4.0, 18.0)), 2) for _ in range(num_records)]
    freq_cardiaca = [round(float(np.clip(np.random.normal(100, 20), 50, 180)), 1) for _ in range(num_records)]
    freq_respiratoria = [round(float(np.clip(np.random.normal(24, 6), 10, 60)), 1) for _ in range(num_records)]
    temperatura = [round(float(np.clip(np.random.normal(38.5, 0.5), 36.5, 41.0)), 1) for _ in range(num_records)]
    behavior_score = [round(float(np.clip(np.random.normal(8, 2), 1, 10)), 1) for _ in range(num_records)]

    status = []
    for ativ, sono_horas, fc, fr, temp, score in zip(atividade_diaria, sono_horas, freq_cardiaca, freq_respiratoria, temperatura, behavior_score):
        if temp > 39.5 or temp < 37.0 or fc > 150 or fr > 45 or score < 4 or ativ < 30:
            st = "RISK"
        elif temp > 39.0 or temp < 37.5 or fc > 120 or fr > 35 or score < 6 or ativ < 50:
            st = "ATTENTION"
        else:
            st = "NORMAL"
        status.append(st)
    
    df = pd.DataFrame({
        'pet_id': pet_ids,
        'data': datas_base,
        'idade_anos': idades_fixas,
        'peso_kg': pesos_variados,
        'atividade_diaria_pct': atividade_diaria,
        'sono_horas': sono_horas, 
        'frequencia_cardiaca_bpm': freq_cardiaca,
        'frequencia_respiratoria': freq_respiratoria,
        'temperatura_c': temperatura,
        'score_comportamento': behavior_score,
        'status': status
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def preprocess_data(raw_filepath: str, processed_filepath: str):
    """Limpa e formata os dados brutos."""
    df = pd.read_csv(raw_filepath)
    
    global_median = df['peso_kg'].median()
    df['peso_kg'] = df.groupby('pet_id')['peso_kg'].transform(lambda x: x.fillna(x.median()))
    df['peso_kg'] = df['peso_kg'].fillna(global_median)
    
    df = df.drop_duplicates(subset=['pet_id', 'data'], keep='last')
    
    mask_1000 = df['pet_id'] == 1000
    if not mask_1000.any() or mask_1000.sum() < 5:
        df = df[df['pet_id'] != 1000]
        linhas_pet_1000 = pd.DataFrame({
            'pet_id': [1000, 1000, 1000, 1000, 1000],
            'data': ['2026-08-01', '2026-08-05', '2026-08-10', '2026-08-15', '2026-08-20'],
            'idade_anos': [3.0, 3.0, 3.0, 3.0, 3.0],
            'peso_kg': [12.5, 12.6, 12.4, 12.5, 12.7],
            'atividade_diaria_pct': [75.0, 80.0, 70.0, 78.0, 82.0],
            'sono_horas': [12.0, 11.5, 12.5, 12.0, 11.8],
            'frequencia_cardiaca_bpm': [95.0, 98.0, 92.0, 96.0, 94.0],
            'frequencia_respiratoria': [22.0, 24.0, 20.0, 23.0, 21.0],
            'temperatura_c': [38.4, 38.5, 38.3, 38.5, 38.4],
            'score_comportamento': [8.5, 9.0, 8.0, 8.8, 9.1],
            'status': ["NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL"]
        })
        df = pd.concat([df, linhas_pet_1000], ignore_index=True)

    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values(by=['pet_id', 'data'])
    df['data'] = df['data'].dt.strftime('%Y-%m-%d')
    
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df.to_csv(processed_filepath, index=False)
    
    print(f"[OK] Dataset processado com histórico salvo em: {processed_filepath}")
    return df

if __name__ == "__main__":
    RAW_PATH = "data/raw/pets_raw_data.csv"
    PROCESSED_PATH = "data/processed/pets_dataset_ready.csv"
    
    generate_synthetic_data(RAW_PATH)
    preprocess_data(RAW_PATH, PROCESSED_PATH)