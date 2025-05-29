import os
import pandas as pd
import numpy as np

def gerar_dataset_1_resumido(data):
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
        pf = pf_valor[0] if len(pf_valor) == 1 else np.nan

        resultados.update({
            'league_name': grupo['league_name'].iloc[0],
            'club_name': club,
            'year': year,
            'PF': pf
        })

        lista_resultados.append(resultados)

    data_1 = pd.DataFrame(lista_resultados)
    data_1 = data_1.sort_values(by=['year', 'PF']).reset_index(drop=True)

    # 🔥 Organizar colunas: league_name, club_name, year, ...resto..., PF
    colunas = data_1.columns.tolist()
    colunas_reordenadas = (
        ['league_name', 'club_name', 'year'] +
        [col for col in colunas if col not in ['league_name', 'club_name', 'year', 'PF']] +
        ['PF']
    )
    data_1 = data_1[colunas_reordenadas]

    return data_1

def gerar_dataset_desempenho_temporal(data, n_anos=1):
    """
    Cria dataset com a posição final no ano atual e no ano + n_anos
    """
    lista_resultados = []
    anos_disponiveis = data['year'].unique()
    ultimo_ano = anos_disponiveis.max()

    for (club, year), grupo in data.groupby(['club_name', 'year']):
        if year + n_anos > ultimo_ano:
            continue  # Não calcula se o ano alvo passa do último ano da base

        rk = grupo['Rk'].iloc[0]
        pf_year = grupo['PF'].iloc[0]
        w = grupo['W'].iloc[0]
        d = grupo['D'].iloc[0]
        l = grupo['L'].iloc[0]
        gf = grupo['GF'].iloc[0]
        ga = grupo['GA'].iloc[0]
        gd = grupo['GD'].iloc[0]
        pts_mp = grupo['Pts/MP'].iloc[0]

        # Buscar PF do ano + n_anos
        grupo_future = data[(data['club_name'] == club) & (data['year'] == year + n_anos)]
        if not grupo_future.empty:
            pf_future = grupo_future['PF'].iloc[0]
        else:
            pf_future = 2  # ✅ Se não está presente, assume faixa inferior

        lista_resultados.append({
            'league_name': grupo['league_name'].iloc[0],
            'club_name': club,
            'year': year,
            'Rk': rk,
            'PF': pf_year,
            'W': w,
            'D': d,
            'L': l,
            'GF': gf,
            'GA': ga,
            'GD': gd,
            'Pts/MP': round(pts_mp, 3),
            f'PF_next_{n_anos}_years': pf_future
        })

    data_result = pd.DataFrame(lista_resultados)
    data_result = data_result.sort_values(by=['year', 'PF']).reset_index(drop=True)

    return data_result

def gerar_dataset_3_combinado(data_1, data_2):
    data_3 = pd.merge(data_1, data_2, on=['league_name', 'club_name', 'year', 'PF'], how='inner')
    data_3 = data_3.sort_values(by=['year', 'PF']).reset_index(drop=True)
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
    data_money.loc[:, 'PF'] = np.select(condicoes, valores)

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

    data_3_1year = gerar_dataset_3_combinado(data_1, data_2_1year)
    data_3_3years = gerar_dataset_3_combinado(data_1, data_2_3years)
    data_3_5years = gerar_dataset_3_combinado(data_1, data_2_5years)

    # Salvar datasets da liga atual
    data_1.to_csv(f'data_1_{league_name}.csv', index=False)
    data_2_1year.to_csv(f'data_2_1year_{league_name}.csv', index=False)
    data_2_3years.to_csv(f'data_2_3years_{league_name}.csv', index=False)
    data_2_5years.to_csv(f'data_2_5years_{league_name}.csv', index=False)

    data_3_1year.to_csv(f'data_3_1year_{league_name}.csv', index=False)
    data_3_3years.to_csv(f'data_3_3years_{league_name}.csv', index=False)
    data_3_5years.to_csv(f'data_3_5years_{league_name}.csv', index=False)

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
data_1_all_leagues.to_csv('data_1_all_leagues.csv', index=False)

data_2_1year_all.to_csv('data_2_1year_all_leagues.csv', index=False)
data_2_3years_all.to_csv('data_2_3years_all_leagues.csv', index=False)
data_2_5years_all.to_csv('data_2_5years_all_leagues.csv', index=False)

data_3_1year_all.to_csv('data_3_1year_all_leagues.csv', index=False)
data_3_3years_all.to_csv('data_3_3years_all_leagues.csv', index=False)
data_3_5years_all.to_csv('data_3_5years_all_leagues.csv', index=False)
