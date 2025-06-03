import os
import pandas as pd
import numpy as np

def gerar_dataset_1_resumido(data):
    """Gera o Dataset 1 com dados de transferências por clube e ano."""
    lista_resultados = []
    for (club, year), grupo in data.groupby(['club_name', 'year']):
        in_transfers = grupo[grupo['transfer_movement'] == 'in']
        out_transfers = grupo[grupo['transfer_movement'] == 'out']

        def calcular_por_posicao(df, pos):
            df_pos = df[df['position'] == pos]
            total_fee = df_pos['fee_cleaned'].sum() if not df_pos.empty else 0
            mean_fee = df_pos['fee_cleaned'].mean() if not df_pos.empty else 0
            mean_age = df_pos['age'].mean() if not df_pos.empty else 0
            count = df_pos.shape[0]
            return total_fee, mean_fee, mean_age, count

        posicoes = ['GK', 'DEF', 'MID', 'STK']
        resultados = {}
        for mov in ['in', 'out']:
            df_mov = in_transfers if mov == 'in' else out_transfers
            total_fee = df_mov['fee_cleaned'].sum() if not df_mov.empty else 0
            mean_fee = df_mov['fee_cleaned'].mean() if not df_mov.empty else 0
            mean_age = df_mov['age'].mean() if not df_mov.empty else 0
            resultados[f'total_fee_{mov}'] = round(total_fee, 3)
            resultados[f'mean_fee_{mov}'] = round(mean_fee, 3)
            resultados[f'mean_age_{mov}'] = round(mean_age, 3)
            for pos in posicoes:
                total_fee_pos, mean_fee_pos, mean_age_pos, count_pos = calcular_por_posicao(df_mov, pos)
                resultados[f'total_fee_{mov}_{pos}'] = round(total_fee_pos, 3)
                resultados[f'mean_fee_{mov}_{pos}'] = round(mean_fee_pos, 3)
                resultados[f'mean_age_{mov}_{pos}'] = round(mean_age_pos, 3)
                resultados[f'count_{pos}_{mov}'] = count_pos

        pf_valor = grupo['PF'].unique()
        pf = pf_valor[0] if len(pf_valor) == 1 else 2  # Default PF=2 se não existir
        resultados.update({
            'league_name': grupo['league_name'].iloc[0],
            'club_name': club,
            'year': year,
            'PF': pf
        })
        lista_resultados.append(resultados)

    data_1 = pd.DataFrame(lista_resultados)
    data_1 = data_1.sort_values(by=['year', 'PF']).reset_index(drop=True)
    colunas = data_1.columns.tolist()
    colunas_reordenadas = (
        ['league_name', 'club_name', 'year'] +
        [col for col in colunas if col not in ['league_name', 'club_name', 'year', 'PF']] +
        ['PF']
    )
    data_1 = data_1[colunas_reordenadas]
    # Substitui NaN por 0 em colunas numéricas (exceto PF)
    colunas_numericas = data_1.select_dtypes(include=['float64', 'int64']).columns
    colunas_numericas = [col for col in colunas_numericas if 'PF' not in col]
    data_1[colunas_numericas] = data_1[colunas_numericas].fillna(0)
    return data_1

def gerar_dataset_desempenho_temporal(data, n_anos=1):
    """Gera o Dataset 2 com dados de performance do ano atual e futuros."""
    lista_resultados = []
    primeiro_ano = 2009
    ultimo_ano = 2019
    valores_padrao = {'PF': 2, 'Rk': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts/MP': 0}

    for (club, year), grupo in data.groupby(['club_name', 'year']):
        if year + n_anos > ultimo_ano:
            continue

        resultado = {
            'league_name': grupo['league_name'].iloc[0],
            'club_name': club,
            'year': year,
        }

        # Dados do ano atual
        for col in ['Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP', 'PF']:
            resultado[col] = grupo[col].iloc[0]

        # Dados de performance dos anos futuros
        for anos_futuros in range(1, n_anos + 1):
            ano_futuro = year + anos_futuros
            if ano_futuro > ultimo_ano:
                for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']:
                    resultado[f'{col}_{anos_futuros}_years_future'] = valores_padrao[col]
            else:
                grupo_futuro = data[(data['club_name'] == club) & (data['year'] == ano_futuro)]
                if not grupo_futuro.empty:
                    for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']:
                        resultado[f'{col}_{anos_futuros}_years_future'] = grupo_futuro[col].iloc[0]
                else:
                    for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']:
                        resultado[f'{col}_{anos_futuros}_years_future'] = valores_padrao[col]

        # Posição final do ano alvo (n_anos à frente)
        grupo_alvo = data[(data['club_name'] == club) & (data['year'] == year + n_anos)]
        resultado[f'PF_next_{n_anos}_years'] = grupo_alvo['PF'].iloc[0] if not grupo_alvo.empty else 2

        lista_resultados.append(resultado)

    data_2 = pd.DataFrame(lista_resultados)
    data_2 = data_2.sort_values(by=['year', 'PF']).reset_index(drop=True)
    # Reorganiza as colunas
    colunas_identificadoras = ['league_name', 'club_name', 'year']
    colunas_atuais = ['Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP', 'PF']
    colunas_futuras = []
    for anos_futuros in range(1, n_anos + 1):
        colunas_futuras.extend([f'{col}_{anos_futuros}_years_future' for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']])
    coluna_alvo = f'PF_next_{n_anos}_years'
    ordem_colunas = colunas_identificadoras + colunas_atuais + colunas_futuras + [coluna_alvo]
    data_2 = data_2[ordem_colunas]
    # Substitui NaN por valores padrão
    colunas_numericas = data_2.select_dtypes(include=['float64', 'int64']).columns
    fill_dict = {col: 2 if 'PF' in col else 0 for col in colunas_numericas}
    data_2 = data_2.fillna(fill_dict)
    return data_2

def gerar_dataset_3_combinado(data_1, data_2, data_transfers, n_anos=1):
    """Gera o Dataset 3 combinando transferências e performance na mesma linha."""
    primeiro_ano = 2009
    ultimo_ano = 2019
    data_3 = data_2.copy()

    # Colunas de transferências do ano atual
    colunas_transferencias = [col for col in data_1.columns if col not in ['league_name', 'club_name', 'year', 'PF']]
    
    # Merge com transferências do ano atual
    data_3 = pd.merge(data_3, data_1, on=['league_name', 'club_name', 'year'], how='inner', suffixes=('', '_transfer'))
    
    # Adiciona transferências dos anos futuros
    for anos_futuros in range(1, n_anos + 1):
        ano_futuro = data_3['year'] + anos_futuros
        data_3[f'temp_year_{anos_futuros}'] = ano_futuro
        colunas_renomeadas = {col: f'{col}_{anos_futuros}_years_future' for col in colunas_transferencias}
        data_1_temp = data_1.copy()
        data_1_temp = data_1_temp[data_1_temp['year'] <= ultimo_ano]
        data_1_temp = data_1_temp.rename(columns=colunas_renomeadas)
        colunas_a_adicionar = ['league_name', 'club_name', 'year'] + list(colunas_renomeadas.values())
        data_1_temp = data_1_temp[colunas_a_adicionar].rename(columns={'year': f'temp_year_{anos_futuros}'})
        data_3 = pd.merge(data_3, data_1_temp, 
                         left_on=['league_name', 'club_name', f'temp_year_{anos_futuros}'], 
                         right_on=['league_name', 'club_name', f'temp_year_{anos_futuros}'], 
                         how='left')
        data_3 = data_3.drop(columns=[f'temp_year_{anos_futuros}'])

        # Preenche valores ausentes nas colunas de transferências
        for col in colunas_renomeadas.values():
            data_3[col] = data_3[col].fillna(0)

    # Reorganiza as colunas
    colunas_identificadoras = ['league_name', 'club_name', 'year']
    colunas_performance_atual = ['Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP', 'PF']
    colunas_transferencias_atual = colunas_transferencias
    colunas_futuras = []
    for anos_futuros in range(1, n_anos + 1):
        colunas_transf_ano = [f'{col}_{anos_futuros}_years_future' for col in colunas_transferencias]
        colunas_perf_ano = [f'{col}_{anos_futuros}_years_future' for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']]
        colunas_futuras.extend(colunas_transf_ano + colunas_perf_ano)
    coluna_alvo = f'PF_next_{n_anos}_years'
    ordem_colunas = colunas_identificadoras + colunas_transferencias_atual + colunas_performance_atual + colunas_futuras + [coluna_alvo]
    ordem_colunas = [col for col in ordem_colunas if col in data_3.columns]
    data_3 = data_3[ordem_colunas]

    # Garante que não haja NaN
    colunas_numericas = data_3.select_dtypes(include=['float64', 'int64']).columns
    fill_dict = {col: 2 if 'PF' in col else 0 for col in colunas_numericas}
    data_3 = data_3.fillna(fill_dict)

    return data_3

# Caminhos e arquivos
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
file_names = [
    'bundesliga_full.csv',
    'la_liga_full.csv',
    'ligue_1_fr_full.csv',
    'premier_league_full.csv',
    'serie_a_it_full.csv'
]

# Listas para os datasets combinados
all_data_1 = []
all_data_2_1year = []
all_data_2_3years = []
all_data_2_5years = []
all_data_3_1year = []
all_data_3_3years = []
all_data_3_5years = []

# Processamento de cada liga
for file_name in file_names:
    league_name = file_name.replace('_full.csv', '')
    file_path = os.path.join(parent_dir, 'data', file_name)
    data = pd.read_csv(file_path)

    # Filtrar anos e valores financeiros
    data_temporal = data[(data['year'] >= 2009) & (data['year'] <= 2019)].copy()
    data_money = data_temporal[(data_temporal['fee_cleaned'].notna()) & (data_temporal['fee_cleaned'] != 0)].copy()

    # Adicionar PF e mapear posições
    condicoes = [
        data_money['Rk'] < 7,
        (data_money['Rk'] >= 7) & (data_money['Rk'] <= 14),
        data_money['Rk'] > 14
    ]
    valores = [0, 1, 2]
    data_money.loc[:, 'PF'] = np.select(condicoes, valores, default=2)

    position_mapping = {
        'GK': 'GK',
        'RB': 'DEF', 'CB': 'DEF', 'LB': 'DEF', 'D': 'DEF',
        'AM': 'MID', 'CM': 'MID', 'DM': 'MID', 'LM': 'MID', 'RM': 'MID',
        'CF': 'STK', 'RW': 'STK', 'LW': 'STK', 'ST': 'STK'
    }
    data_money.loc[:, 'position'] = data_money['position'].replace(position_mapping)

    # Gerar datasets individuais
    data_1 = gerar_dataset_1_resumido(data_money)
    data_2_1year = gerar_dataset_desempenho_temporal(data_money, n_anos=1)
    data_2_3years = gerar_dataset_desempenho_temporal(data_money, n_anos=3)
    data_2_5years = gerar_dataset_desempenho_temporal(data_money, n_anos=5)

    data_3_1year = gerar_dataset_3_combinado(data_1, data_2_1year, data_money, n_anos=1)
    data_3_3years = gerar_dataset_3_combinado(data_1, data_2_3years, data_money, n_anos=3)
    data_3_5years = gerar_dataset_3_combinado(data_1, data_2_5years, data_money, n_anos=5)

    # Salvar datasets da liga atual
    os.makedirs('results', exist_ok=True)
    data_1.to_csv(f'results/data_1_{league_name}.csv', index=False)
    data_2_1year.to_csv(f'results/data_2_1year_{league_name}.csv', index=False)
    data_2_3years.to_csv(f'results/data_2_3years_{league_name}.csv', index=False)
    data_2_5years.to_csv(f'results/data_2_5years_{league_name}.csv', index=False)
    data_3_1year.to_csv(f'results/data_3_1year_{league_name}.csv', index=False)
    data_3_3years.to_csv(f'results/data_3_3years_{league_name}.csv', index=False)
    data_3_5years.to_csv(f'results/data_3_5years_{league_name}.csv', index=False)

    # Armazenar para os datasets combinados
    all_data_1.append(data_1)
    all_data_2_1year.append(data_2_1year)
    all_data_2_3years.append(data_2_3years)
    all_data_2_5years.append(data_2_5years)
    all_data_3_1year.append(data_3_1year)
    all_data_3_3years.append(data_3_3years)
    all_data_3_5years.append(data_3_5years)

# Gerar e salvar datasets combinados (todas as ligas)
data_1_all_leagues = pd.concat(all_data_1, ignore_index=True)
data_2_1year_all = pd.concat(all_data_2_1year, ignore_index=True)
data_2_3years_all = pd.concat(all_data_2_3years, ignore_index=True)
data_2_5years_all = pd.concat(all_data_2_5years, ignore_index=True)
data_3_1year_all = pd.concat(all_data_3_1year, ignore_index=True)
data_3_3years_all = pd.concat(all_data_3_3years, ignore_index=True)
data_3_5years_all = pd.concat(all_data_3_5years, ignore_index=True)

# Salvar CSVs combinados
os.makedirs('results', exist_ok=True)
data_1_all_leagues.to_csv('results/data_1_all_leagues.csv', index=False)
data_2_1year_all.to_csv('results/data_2_1year_all_leagues.csv', index=False)
data_2_3years_all.to_csv('results/data_2_3years_all_leagues.csv', index=False)
data_2_5years_all.to_csv('results/data_2_5years_all_leagues.csv', index=False)
data_3_1year_all.to_csv('results/data_3_1year_all_leagues.csv', index=False)
data_3_3years_all.to_csv('results/data_3_3years_all_leagues.csv', index=False)
data_3_5years_all.to_csv('results/data_3_5years_all_leagues.csv', index=False)

print("Datasets gerados e salvos com sucesso!")