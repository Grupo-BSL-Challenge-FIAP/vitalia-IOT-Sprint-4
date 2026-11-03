import os
import oracledb
from dotenv import load_dotenv
from api.exceptions import DatabaseConnectionError

load_dotenv()

class PetRepository:
    def __init__(self):
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")

    def _get_connection(self):
        try:
            return oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
        except Exception as e:
            raise DatabaseConnectionError(f"Erro de conexão com a base de dados Oracle: {e}")

    def obter_informacoes_pet(self, pet_id: int):
        query = """
            SELECT p.PET_ID, p.NAME, p.BIRTH_DATE, p.BREED_ID, p.WEIGHT_KG, p.SEX, p.STATUS, s.NAME AS SPECIES_NAME
            FROM TB_VITALIA_PET p
            LEFT JOIN TB_VITALIA_BREED b ON p.BREED_ID = b.BREED_ID
            LEFT JOIN TB_VITALIA_SPECIES s ON b.SPECIES_ID = s.SPECIES_ID
            WHERE p.PET_ID = :pet_id
        """

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, pet_id=pet_id)
                row = cursor.fetchone()
                if row:
                    return {
                        "pet_id": row[0],
                        "PET_ID": row[0],
                        "nome": row[1],
                        "NOME": row[1],
                        "birth_date": row[2],
                        "BIRTH_DATE": row[2],
                        "breed_id": row[3],
                        "BREED_ID": row[3],
                        "weight_kg": row[4],
                        "WEIGHT_KG": row[4],
                        "sex": row[5],
                        "SEX": row[5],
                        "status": row[6],
                        "STATUS": row[6],
                        "especie": row[7] if len(row) > 7 else "Não informada",
                        "ESPECIE": row[7] if len(row) > 7 else "Não informada",
                    }
        return None 

    def obter_historico_medicoes(self, pet_id: int):
        query = """
            SELECT MEASUREMENT_ID, PET_ID, ACTIVITY_LEVEL, SLEEP_HOURS, 
                   HEART_RATE_BPM, RESPIRATORY_RATE, TEMPERATURE_C, 
                   BEHAVIOR_SCORE, MEASUREMENT_DATE
            FROM TB_VITALIA_HEALTH_MEASUREMENT
            WHERE PET_ID = :pet_id
            ORDER BY MEASUREMENT_DATE DESC
        """
        historico = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, pet_id=pet_id)
                columns = [col[0].lower() for col in cursor.description]
                for row in cursor.fetchall():
                    historico.append(dict(zip(columns, row)))
        return historico