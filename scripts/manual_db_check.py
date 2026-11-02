import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from api.repositories.pet_repository import PetRepository
from api.services.history_service import HistoryService

print('--- A testar informacoes do pet 1 ---')
repo = PetRepository()
pet_info = repo.obter_informacoes_pet(1)
print(f'Pet Info: {pet_info}')

print('\n--- A testar historico de medicoes ---')
medicoes = repo.obter_historico_medicoes(1)
print(f'Medicoes encontradas: {len(medicoes)}')
for m in medicoes:
    print(m)

print('\n--- A testar HistoryService.obter_ultimo_registro ---')
service = HistoryService(repo)
ultimo = service.obter_ultimo_registro(1)
print(f'Resultado final do servico: {ultimo}')
