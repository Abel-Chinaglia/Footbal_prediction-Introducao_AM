import os
import pandas as pd
import numpy as np

# Caminho do diretório atual
current_dir = os.getcwd()

# Voltar uma pasta (diretório pai)
parent_dir = os.path.dirname(current_dir)

# Caminho completo até a pasta 'data' e o arquivo CSV
file_path = os.path.join(parent_dir, 'data', 'premier_league_full.csv')

# Ler o arquivo CSV
data = pd.read_csv(file_path)

# Filtrar anos entre 2009 e 2019 
data_temporal = data[(data['year'] >= 2009) & (data['year'] <= 2019)]

# Manter somente valores financeiros diferentes de 0 e não nulos
data_money = data_temporal[(data_temporal['fee_cleaned'].notna()) & (data_temporal['fee_cleaned'] != 0)]

#Criando variáveis categóricas da posição final do time, sendo elas colocadas na coluna PF (posição final)
# Definir as condições
condicoes = [
    data_money['Rk'] < 7,
    (data_money['Rk'] >= 7) & (data_money['Rk'] <= 14),
    data_money['Rk'] > 14
]

# Definir os valores correspondentes
valores = [0, 1, 2]

# Criar a nova coluna 'PF'
data_money['PF'] = np.select(condicoes, valores)

#Criando banco de dados 1 -  transferencia de jogadores -> posição final
data_1 = data_money[['age','transfer_period','fee_cleaned','transfer_movement','PF']]

#Criando banco de dados 2 -  Desempenho temporada anterior -> posição final
data_2 = data_money[['Rk','W','D','L','GF','GA','GD','Pts/MP','PF']]

#Criando banco de dados 3 -  transferencia de jogadores + desempenho anterior -> posição final
data_3 = data_money[['age','transfer_period','fee_cleaned','transfer_movement',
                     'Rk','W','D','L','GF','GA','GD','Pts/MP','PF']]

