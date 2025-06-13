# =============================================================================
# generate_football_datasets.py
# =============================================================================
# Autor: Abel Gonçalves Chinaglia
# Doutorando no PPGRDF - FMRP - USP
# Data: 5 jun. 2025
# Versão do Python: 3.11

# Descrição:
# ----------
# Este script processa dados brutos de clubes de futebol das principais ligas europeias
# e gera três tipos de datasets prontos para análise e modelagem:
# (1) dados de transferências, (2) dados de desempenho esportivo e (3) combinação de ambos.
# Para cada tipo, o script cria janelas temporais de 1, 3 e 5 anos, incluindo dados do
# ano atual e projeções de anos futuros.

# Funcionalidades Principais:
# ----------------------------
# - Lê arquivos CSV de cinco ligas: Bundesliga, La Liga, Ligue 1, Premier League e Serie A
# - Calcula métricas agregadas de transferências (valores, idades e posições)
# - Organiza dados de desempenho por temporada (vitórias, gols, ranking etc.)
# - Gera datasets com janelas temporais de 1, 3 e 5 anos
# - Adiciona colunas de previsão (anos futuros) e variável alvo (PF futura)
# - Combina dados de transferências e desempenho em um terceiro dataset
# - Salva todos os datasets gerados em formato CSV no diretório `results/`

# Execução:
# ---------
# - Verifique se os arquivos CSV de entrada estão em: ../data/
#   Arquivos esperados:
#     * bundesliga_full.csv
#     * la_liga_full.csv
#     * ligue_1_fr_full.csv
#     * premier_league_full.csv
#     * serie_a_it_full.csv
# - Execute o script:
#   $ python generate_football_datasets.py
# - Os arquivos gerados serão salvos em:
#   ./results/

# Estrutura dos Datasets Gerados:
# -------------------------------
# Dataset 1 (transferências): dados por clube e ano com projeções futuras
# Dataset 2 (desempenho): métricas esportivas com projeções e PF futura como alvo
# Dataset 3 (combinado): junção dos datasets 1 e 2 para análises integradas
# - Para cada dataset: versões com janelas de 1, 3 e 5 anos
# - Arquivos por liga e versões agregadas com todas as ligas

# Licença:
# --------
# Este programa está licenciado sob a GNU Lesser General Public License v3.0.
# Para mais informações, acesse: https://www.gnu.org/licenses/lgpl-3.0.html

# =============================================================================

import os
import pandas as pd
import numpy as np

def gerar_dataset_1_resumido(data, n_anos=1):
    """Gera o Dataset 1 com dados de transferências do ano atual e anos futuros."""
    lista_resultados = []
    primeiro_ano = 2009
    ultimo_ano = 2019
    valores_padrao = {}
    for mov in ['in', 'out']:
        for pos in ['GK', 'DEF', 'MID', 'STK']:
            valores_padrao.update({
                f'total_fee_{mov}': 0,
                f'mean_fee_{mov}': 0,
                f'mean_age_{mov}': 0,
                f'total_fee_{mov}_{pos}': 0,
                f'mean_fee_{mov}_{pos}': 0,
                f'mean_age_{mov}_{pos}': 0,
                f'count_{pos}_{mov}': 0
            })

    for (club, year), grupo in data.groupby(['club_name', 'year']):
        # Verifica se o clube tem dados em todos os anos futuros necessários
        has_all_future_data = True
        for anos_futuros in range(n_anos + 1):  # Inclui ano atual (0) até n_anos
            ano_futuro = year + anos_futuros
            if ano_futuro > ultimo_ano:
                has_all_future_data = False
                break
            grupo_futuro = data[(data['club_name'] == club) & (data['year'] == ano_futuro)]
            if grupo_futuro.empty:
                has_all_future_data = False
                break
        if not has_all_future_data:
            continue

        resultado = {
            'league_name': grupo['league_name'].iloc[0],
            'club_name': club,
            'year': year,
        }

        def calcular_por_posicao(df, pos):
            df_pos = df[df['position'] == pos]
            total_fee = df_pos['fee_cleaned'].sum() if not df_pos.empty else 0
            mean_fee = df_pos['fee_cleaned'].mean() if not df_pos.empty else 0
            mean_age = df_pos['age'].mean() if not df_pos.empty else 0
            count = df_pos.shape[0]
            return total_fee, mean_fee, mean_age, count

        posicoes = ['GK', 'DEF', 'MID', 'STK']

        # Dados do ano atual
        in_transfers = grupo[grupo['transfer_movement'] == 'in']
        out_transfers = grupo[grupo['transfer_movement'] == 'out']
        for mov in ['in', 'out']:
            df_mov = in_transfers if mov == 'in' else out_transfers
            total_fee = df_mov['fee_cleaned'].sum() if not df_mov.empty else 0
            mean_fee = df_mov['fee_cleaned'].mean() if not df_mov.empty else 0
            mean_age = df_mov['age'].mean() if not df_mov.empty else 0
            resultado[f'total_fee_{mov}'] = round(total_fee, 3)
            resultado[f'mean_fee_{mov}'] = round(mean_fee, 3)
            resultado[f'mean_age_{mov}'] = round(mean_age, 3)
            for pos in posicoes:
                total_fee_pos, mean_fee_pos, mean_age_pos, count_pos = calcular_por_posicao(df_mov, pos)
                resultado[f'total_fee_{mov}_{pos}'] = round(total_fee_pos, 3)
                resultado[f'mean_fee_{mov}_{pos}'] = round(mean_fee_pos, 3)
                resultado[f'mean_age_{mov}_{pos}'] = round(mean_age_pos, 3)
                resultado[f'count_{pos}_{mov}'] = count_pos

        # Dados dos anos futuros
        for anos_futuros in range(1, n_anos + 1):
            ano_futuro = year + anos_futuros
            grupo_futuro = data[(data['club_name'] == club) & (data['year'] == ano_futuro)]
            in_transfers_futuro = grupo_futuro[grupo_futuro['transfer_movement'] == 'in']
            out_transfers_futuro = grupo_futuro[grupo_futuro['transfer_movement'] == 'out']
            for mov in ['in', 'out']:
                df_mov = in_transfers_futuro if mov == 'in' else out_transfers_futuro
                total_fee = df_mov['fee_cleaned'].sum() if not df_mov.empty else 0
                mean_fee = df_mov['fee_cleaned'].mean() if not df_mov.empty else 0
                mean_age = df_mov['age'].mean() if not df_mov.empty else 0
                resultado[f'total_fee_{mov}_{anos_futuros}_years_future'] = round(total_fee, 3)
                resultado[f'mean_fee_{mov}_{anos_futuros}_years_future'] = round(mean_fee, 3)
                resultado[f'mean_age_{mov}_{anos_futuros}_years_future'] = round(mean_age, 3)
                for pos in posicoes:
                    total_fee_pos, mean_fee_pos, mean_age_pos, count_pos = calcular_por_posicao(df_mov, pos)
                    resultado[f'total_fee_{mov}_{pos}_{anos_futuros}_years_future'] = round(total_fee_pos, 3)
                    resultado[f'mean_fee_{mov}_{pos}_{anos_futuros}_years_future'] = round(mean_fee_pos, 3)
                    resultado[f'mean_age_{mov}_{pos}_{anos_futuros}_years_future'] = round(mean_age_pos, 3)
                    resultado[f'count_{pos}_{mov}_{anos_futuros}_years_future'] = count_pos

        pf_valor = grupo['PF'].unique()
        resultado['PF'] = pf_valor[0] if len(pf_valor) == 1 else 2
        lista_resultados.append(resultado)

    data_1 = pd.DataFrame(lista_resultados)
    if data_1.empty:
        return data_1
    data_1 = data_1.sort_values(by=['year', 'PF']).reset_index(drop=True)
    colunas_identificadoras = ['league_name', 'club_name', 'year']
    colunas_atuais = [col for col in data_1.columns if col not in ['league_name', 'club_name', 'year', 'PF'] and '_years_future' not in col]
    colunas_futuras = []
    for anos_futuros in range(1, n_anos + 1):
        colunas_futuras.extend([f'{col}_{anos_futuros}_years_future' for col in colunas_atuais])
    ordem_colunas = colunas_identificadoras + colunas_atuais + colunas_futuras + ['PF']
    data_1 = data_1[ordem_colunas]
    colunas_numericas = data_1.select_dtypes(include=['float64', 'int64']).columns
    colunas_numericas = [col for col in colunas_numericas if 'PF' not in col]
    data_1[colunas_numericas] = data_1[colunas_numericas].fillna(0)
    return data_1

def gerar_dataset_desempenho_temporal(data, n_anos=1):
    """Gera o Dataset 2 com dados de performance do ano atual e, para n_anos=3 ou 5, anos futuros intermediários."""
    lista_resultados = []
    primeiro_ano = 2009
    ultimo_ano = 2019
    valores_padrao = {'PF': 2, 'Rk': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts/MP': 0}

    for (club, year), grupo in data.groupby(['club_name', 'year']):
        # Verifica se o clube tem dados em todos os anos futuros necessários
        has_all_future_data = True
        for anos_futuros in range(n_anos + 1):  # Inclui ano atual (0) até n_anos
            ano_futuro = year + anos_futuros
            if ano_futuro > ultimo_ano:
                has_all_future_data = False
                break
            grupo_futuro = data[(data['club_name'] == club) & (data['year'] == ano_futuro)]
            if grupo_futuro.empty:
                has_all_future_data = False
                break
        if not has_all_future_data:
            continue

        resultado = {
            'league_name': grupo['league_name'].iloc[0],
            'club_name': club,
            'year': year,
        }

        # Dados do ano atual
        for col in ['Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP', 'PF']:
            resultado[col] = grupo[col].iloc[0]

        # Para n_anos=1, não inclui dados futuros intermediários
        if n_anos > 1:
            # Dados de performance dos anos futuros intermediários (até n_anos - 1)
            for anos_futuros in range(1, n_anos):
                ano_futuro = year + anos_futuros
                grupo_futuro = data[(data['club_name'] == club) & (data['year'] == ano_futuro)]
                for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']:
                    resultado[f'{col}_{anos_futuros}_years_future'] = grupo_futuro[col].iloc[0]

        # Posição final do ano alvo (n_anos à frente)
        grupo_alvo = data[(data['club_name'] == club) & (data['year'] == year + n_anos)]
        resultado[f'PF_next_{n_anos}_years'] = grupo_alvo['PF'].iloc[0]

        lista_resultados.append(resultado)

    data_2 = pd.DataFrame(lista_resultados)
    if data_2.empty:
        return data_2
    data_2 = data_2.sort_values(by=['year', 'PF']).reset_index(drop=True)
    colunas_identificadoras = ['league_name', 'club_name', 'year']
    colunas_atuais = ['Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP', 'PF']
    colunas_futuras = []
    if n_anos > 1:
        for anos_futuros in range(1, n_anos):
            colunas_futuras.extend([f'{col}_{anos_futuros}_years_future' for col in ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']])
    coluna_alvo = f'PF_next_{n_anos}_years'
    ordem_colunas = colunas_identificadoras + colunas_atuais + colunas_futuras + [coluna_alvo]
    data_2 = data_2[ordem_colunas]
    colunas_numericas = data_2.select_dtypes(include=['float64', 'int64']).columns
    fill_dict = {col: 2 if 'PF' in col else 0 for col in colunas_numericas}
    data_2 = data_2.fillna(fill_dict)
    return data_2

def gerar_dataset_3_combinado(data_1, data_2, n_anos=1):
    """Gera o Dataset 3 combinando transferências e performance."""
    data_3 = pd.merge(data_1, data_2, on=['league_name', 'club_name', 'year', 'PF'], how='inner')
    colunas_identificadoras = ['league_name', 'club_name', 'year']
    colunas_transferencias = [col for col in data_1.columns if col not in ['league_name', 'club_name', 'year', 'PF']]
    colunas_performance = [col for col in data_2.columns if col not in ['league_name', 'club_name', 'year', f'PF_next_{n_anos}_years']]
    coluna_alvo = f'PF_next_{n_anos}_years'
    ordem_colunas = colunas_identificadoras + colunas_transferencias + colunas_performance + [coluna_alvo]
    data_3 = data_3[ordem_colunas]
    data_3 = data_3.sort_values(by=['year', 'PF']).reset_index(drop=True)
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

all_data_1_1year = []
all_data_1_3years = []
all_data_1_5years = []
all_data_2_1year = []
all_data_2_3years = []
all_data_2_5years = []
all_data_3_1year = []
all_data_3_3years = []
all_data_3_5years = []

for file_name in file_names:
    league_name = file_name.replace('_full.csv', '')
    file_path = os.path.join(parent_dir, 'data', file_name)
    data = pd.read_csv(file_path)

    data_temporal = data[(data['year'] >= 2009) & (data['year'] <= 2019)].copy()
    data_money = data_temporal[(data_temporal['fee_cleaned'].notna()) & (data_temporal['fee_cleaned'] != 0)].copy()

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

    data_1_1year = gerar_dataset_1_resumido(data_money, n_anos=1)
    data_1_3years = gerar_dataset_1_resumido(data_money, n_anos=3)
    data_1_5years = gerar_dataset_1_resumido(data_money, n_anos=5)
    data_2_1year = gerar_dataset_desempenho_temporal(data_money, n_anos=1)
    data_2_3years = gerar_dataset_desempenho_temporal(data_money, n_anos=3)
    data_2_5years = gerar_dataset_desempenho_temporal(data_money, n_anos=5)

    data_3_1year = gerar_dataset_3_combinado(data_1_1year, data_2_1year, n_anos=1)
    data_3_3years = gerar_dataset_3_combinado(data_1_3years, data_2_3years, n_anos=3)
    data_3_5years = gerar_dataset_3_combinado(data_1_5years, data_2_5years, n_anos=5)

    os.makedirs('results', exist_ok=True)
    data_1_1year.to_csv(f'results/data_1_1year_{league_name}.csv', index=False)
    data_1_3years.to_csv(f'results/data_1_3years_{league_name}.csv', index=False)
    data_1_5years.to_csv(f'results/data_1_5years_{league_name}.csv', index=False)
    data_2_1year.to_csv(f'results/data_2_1year_{league_name}.csv', index=False)
    data_2_3years.to_csv(f'results/data_2_3years_{league_name}.csv', index=False)
    data_2_5years.to_csv(f'results/data_2_5years_{league_name}.csv', index=False)
    data_3_1year.to_csv(f'results/data_3_1year_{league_name}.csv', index=False)
    data_3_3years.to_csv(f'results/data_3_3years_{league_name}.csv', index=False)
    data_3_5years.to_csv(f'results/data_3_5years_{league_name}.csv', index=False)

    all_data_1_1year.append(data_1_1year)
    all_data_1_3years.append(data_1_3years)
    all_data_1_5years.append(data_1_5years)
    all_data_2_1year.append(data_2_1year)
    all_data_2_3years.append(data_2_3years)
    all_data_2_5years.append(data_2_5years)
    all_data_3_1year.append(data_3_1year)
    all_data_3_3years.append(data_3_3years)
    all_data_3_5years.append(data_3_5years)

data_1_1year_all = pd.concat(all_data_1_1year, ignore_index=True)
data_1_3years_all = pd.concat(all_data_1_3years, ignore_index=True)
data_1_5years_all = pd.concat(all_data_1_5years, ignore_index=True)
data_2_1year_all = pd.concat(all_data_2_1year, ignore_index=True)
data_2_3years_all = pd.concat(all_data_2_3years, ignore_index=True)
data_2_5years_all = pd.concat(all_data_2_5years, ignore_index=True)
data_3_1year_all = pd.concat(all_data_3_1year, ignore_index=True)
data_3_3years_all = pd.concat(all_data_3_3years, ignore_index=True)
data_3_5years_all = pd.concat(all_data_3_5years, ignore_index=True)

os.makedirs('results', exist_ok=True)
data_1_1year_all.to_csv('results/data_1_1year_all_leagues.csv', index=False)
data_1_3years_all.to_csv('results/data_1_3years_all_leagues.csv', index=False)
data_1_5years_all.to_csv('results/data_1_5years_all_leagues.csv', index=False)
data_2_1year_all.to_csv('results/data_2_1year_all_leagues.csv', index=False)
data_2_3years_all.to_csv('results/data_2_3years_all_leagues.csv', index=False)
data_2_5years_all.to_csv('results/data_2_5years_all_leagues.csv', index=False)
data_3_1year_all.to_csv('results/data_3_1year_all_leagues.csv', index=False)
data_3_3years_all.to_csv('results/data_3_3years_all_leagues.csv', index=False)
data_3_5years_all.to_csv('results/data_3_5years_all_leagues.csv', index=False)

print("Datasets gerados e salvos com sucesso!")
