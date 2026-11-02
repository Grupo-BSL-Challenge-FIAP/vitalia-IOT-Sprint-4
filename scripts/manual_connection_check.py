from dotenv import load_dotenv
load_dotenv() 

from api.repositories.pet_repository import PetRepository

repo = PetRepository()
print("A testar a conexão com a base de dados Oracle...")

pet_info = repo.obter_informacoes_pet(1)
print("Informações do pet:", pet_info)

historico = repo.obter_historico_medicoes(1)
print(f"Histórico obtido com sucesso! Total de registos: {len(historico)}")