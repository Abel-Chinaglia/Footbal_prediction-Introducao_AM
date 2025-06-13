# =============================================================================
# generate_repeated_football_datasets.py
# =============================================================================
# Autor: Abel Gonçalves Chinaglia
# Doutorando no PPGRDF - FMRP - USP
# Data: 5 jun. 2025
# Versão do Python: 3.11

# Descrição:
# ----------
# Este script gera bancos de dados estruturados a partir de dados brutos de clubes
# de futebol das principais ligas europeias. A partir dos dados de transferências e
# desempenho esportivo, o script cria três conjuntos de dados (transferências, desempenho
# e combinado) com janelas temporais de 1, 3 e 5 anos, incluindo projeções para anos futuros.

# Funcionalidades Principais:
# ----------------------------
# - Processa dados de cinco ligas europeias (Bundesliga, La Liga, Ligue 1, Premier League e Serie A)
# - Agrupa e calcula métricas de transferências por posição e tipo (entrada/saída)
# - Calcula métricas de desempenho esportivo por temporada
# - Gera datasets com dados atuais e projeções para anos futuros (1, 3 e 5 anos)
# - Implementa lógica para preencher valores faltantes com dados do ano anterior mais próximo
# - Salva os datasets formatados em arquivos CSV organizados por liga e por tipo de dataset

# Execução:
# ---------
# - Certifique-se de que os arquivos de entrada estejam no diretório: /data/
#   Arquivos esperados: bundesliga_full.csv, la_liga_full.csv, ligue_1_fr_full.csv,
#   premier_league_full.csv, serie_a_it_full.csv
# - Execute o script com:
#   $ python generate_repeated_football_datasets.py
# - Os arquivos gerados serão salvos no diretório: /data/pre_process_repeated/

# Estrutura dos Datasets Gerados:
# -------------------------------
# Dataset 1: Dados de transferências com projeções futuras
# Dataset 2: Dados de desempenho com projeções e alvo (PF em n anos)
# Dataset 3: Combinação dos dados de transferências e desempenho
# - Para cada dataset: versões com janelas de 1, 3 e 5 anos
# - Arquivos CSV separados por liga e arquivos combinados com todas as ligas

# Formato dos Arquivos de Entrada:
# --------------------------------
# - Dados brutos com colunas como: league_name, club_name, year, Rk, W, D, L, GF, GA, fee_cleaned, position, etc.
# - Os valores de PF (Performance Final) são atribuídos com base no ranking da temporada

# Licença:
# --------
# Este programa está licenciado sob a GNU Lesser General Public License v3.0.
# Para mais informações, acesse: https://www.gnu.org/licenses/lgpl-3.0.html

# =============================================================================

import os
import pandas as pd
import numpy as np

def get_previous_non_default_data(club, ano_futuro, data, ano_base, colunas, valores_padrao):
    """
    Busca dados do ano imediatamente anterior com valores não padrão.
    
    Args:
        club (str): Nome do clube.
        ano_futuro (int): Ano futuro a ser preenchido.
        data (pd.DataFrame): DataFrame com os dados (data_1_base ou data_2_base).
        ano_base (int): Ano base da linha atual.
        colunas (list): Colunas a serem retornadas (base, sem sufixo _years_future).
        valores_padrao (dict): Valores padrão para comparação.
    
    Returns:
        dict: Dados do ano anterior mais recente não padrão ou None se não encontrado.
    """
    # Tenta o ano futuro diretamente
    grupo_futuro = data[(data['club_name'] == club) & (data['year'] == ano_futuro)]
    if not grupo_futuro.empty:
        return grupo_futuro[colunas].iloc[0].to_dict()
    
    # Tenta o ano base
    grupo_base = data[(data['club_name'] == club) & (data['year'] == ano_base)]
    if not grupo_base.empty:
        valores = grupo_base[colunas].iloc[0].to_dict()
        if not all(valores[col] == valores_padrao.get(col, 0) for col in colunas):
            return valores
    
    # Tenta anos anteriores ao ano base
    ano_anterior = ano_base - 1
    while ano_anterior >= 2009:
        grupo_anterior = data[(data['club_name'] == club) & (data['year'] == ano_anterior)]
        if not grupo_anterior.empty:
            valores = grupo_anterior[colunas].iloc[0].to_dict()
            if not all(valores[col] == valores_padrao.get(col, 0) for col in colunas):
                return valores
        ano_anterior -= 1
    return None

def gerar_data_1_base(data_money):
    """Gera o DataFrame base para Dataset 1 com métricas de transferências por ano."""
    lista_resultados = []
    for (club, year), grupo in data_money.groupby(['club_name', 'year']):
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
        for mov in ['in', 'out']:
            df_mov = grupo[grupo['transfer_movement'] == mov]
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
        resultado['PF'] = grupo['PF'].iloc[0]
        lista_resultados.append(resultado)
    data_1_base = pd.DataFrame(lista_resultados)
    data_1_base = data_1_base.sort_values(by=['year', 'PF']).reset_index(drop=True)
    return data_1_base

def gerar_data_2_base(data_temporal):
    """Gera o DataFrame base para Dataset 2 com métricas de desempenho por ano."""
    data_2_base = data_temporal[['league_name', 'club_name', 'year', 'PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']].drop_duplicates().reset_index(drop=True)
    return data_2_base

def gerar_dataset_1_resumido(data_1_base, n_anos=1):
    """Gera o Dataset 1 com dados de transferências do ano atual e anos futuros."""
    lista_resultados = []
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
    colunas_transferencias = list(valores_padrao.keys())
    
    for idx, row in data_1_base.iterrows():
        club = row['club_name']
        year = row['year']
        if year + n_anos > ultimo_ano:
            continue
        resultado = {
            'league_name': row['league_name'],
            'club_name': club,
            'year': year,
        }
        for col in colunas_transferencias:
            resultado[col] = row[col]
        for anos_futuros in range(1, n_anos + 1):
            ano_futuro = year + anos_futuros
            previous_data = get_previous_non_default_data(club, ano_futuro, data_1_base, year, colunas_transferencias, valores_padrao)
            if previous_data:
                for col in colunas_transferencias:
                    resultado[f'{col}_{anos_futuros}_years_future'] = round(previous_data.get(col, valores_padrao[col]), 3)
            else:
                for col in colunas_transferencias:
                    resultado[f'{col}_{anos_futuros}_years_future'] = valores_padrao[col]
        resultado['PF'] = row['PF']
        lista_resultados.append(resultado)
    
    data_1 = pd.DataFrame(lista_resultados)
    if data_1.empty:
        return data_1
    
    data_1 = data_1.sort_values(by=['year', 'PF']).reset_index(drop=True)
    colunas_identificadores = ['league_name', 'club_name', 'year']
    colunas_atuais = [col for col in data_1.columns if col not in ['league_name', 'club_name', 'year', 'PF'] and '_years_future' not in col]
    colunas_futuras = []
    for anos_futuros in range(1, n_anos + 1):
        colunas_futuras.extend([f'{col}_{anos_futuros}_years_future' for col in colunas_atuais])
    ordem_colunas = colunas_identificadores + colunas_atuais + colunas_futuras + ['PF']
    data_1 = data_1[ordem_colunas]
    colunas_numericas = data_1.select_dtypes(include=['float64', 'int64']).columns
    colunas_numericas = [col for col in colunas_numericas if col != 'PF']
    data_1[colunas_numericas] = data_1[colunas_numericas].fillna(0)
    return data_1

def gerar_dataset_desempenho_temporal(data_2_base, n_anos=1):
    """Gera o Dataset 2 com dados de desempenho do ano atual e anos futuros intermediários."""
    lista_resultados = []
    ultimo_ano = 2019
    valores_padrao = {'PF': 2, 'Rk': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts/MP': 0}
    colunas_desempenho = ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']
    
    for idx, row in data_2_base.iterrows():
        club = row['club_name']
        year = row['year']
        if year + n_anos > ultimo_ano:
            continue
        resultado = {
            'league_name': row['league_name'],
            'club_name': club,
            'year': year,
        }
        for col in colunas_desempenho:
            resultado[col] = row[col]
        if n_anos > 1:
            for anos_futuros in range(1, n_anos):
                ano_futuro = year + anos_futuros
                previous_data = get_previous_non_default_data(club, ano_futuro, data_2_base, year, colunas_desempenho, valores_padrao)
                if previous_data:
                    for col in colunas_desempenho:
                        resultado[f'{col}_{anos_futuros}_years_future'] = previous_data[col]
                else:
                    for col in colunas_desempenho:
                        resultado[f'{col}_{anos_futuros}_years_future'] = valores_padrao[col]
        # Último ano (alvo)
        ano_alvo = year + n_anos
        previous_data = get_previous_non_default_data(club, ano_alvo, data_2_base, year, colunas_desempenho, valores_padrao)
        resultado[f'PF_next_{n_anos}_years'] = previous_data['PF'] if previous_data else valores_padrao['PF']
        lista_resultados.append(resultado)
    
    data_2 = pd.DataFrame(lista_resultados)
    if data_2.empty:
        return data_2
    
    data_2 = data_2.sort_values(by=['year', 'PF']).reset_index(drop=True)
    colunas_identificadores = ['league_name', 'club_name', 'year']
    colunas_atuais = ['Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP', 'PF']
    colunas_futuras = []
    if n_anos > 1:
        for anos_futuros in range(1, n_anos):
            colunas_futuras.extend([f'{col}_{anos_futuros}_years_future' for col in colunas_desempenho])
    coluna_alvo = f'PF_next_{n_anos}_years'
    ordem_colunas = colunas_identificadores + colunas_atuais + colunas_futuras + [coluna_alvo]
    data_2 = data_2[ordem_colunas]
    colunas_numericas = data_2.select_dtypes(include=['float64', 'int64']).columns
    fill_dict = {col: 2 if 'PF' in col else 0 for col in colunas_numericas}
    data_2 = data_2.fillna(fill_dict)
    return data_2

def gerar_dataset_3_combinado(data_1, data_2, n_anos=1):
    """Gera o Dataset 3 combinando transferências e desempenho."""
    data_3 = pd.merge(data_1, data_2, on=['league_name', 'club_name', 'year', 'PF'], how='inner')
    colunas_identificadores = ['league_name', 'club_name', 'year']
    colunas_transferencias = [col for col in data_1.columns if col not in ['league_name', 'club_name', 'year', 'PF']]
    colunas_performance = [col for col in data_2.columns if col not in ['league_name', 'club_name', 'year', f'PF_next_{n_anos}_years']]
    coluna_alvo = f'PF_next_{n_anos}_years'
    ordem_colunas = colunas_identificadores + colunas_transferencias + colunas_performance + [coluna_alvo]
    data_3 = data_3[ordem_colunas]
    data_3 = data_3.sort_values(by=['year', 'PF']).reset_index(drop=True)
    colunas_numericas = data_3.select_dtypes(include=['float64', 'int64']).columns
    fill_dict = {col: 2 if 'PF' in col else 0 for col in colunas_numericas}
    data_3 = data_3.fillna(fill_dict)
    return data_3

def main():
    # Caminhos e arquivos
    script_dir = os.getcwd()  # Diretório atual (scripts)
    project_dir = os.path.dirname(script_dir)  # Diretório raiz do projeto
    data_dir = os.path.join(project_dir, 'data')  # Pasta data no diretório raiz
    output_dir = os.path.join(data_dir, 'pre_process_repeated')
    
    # Cria a pasta pre_process_repeated, se não existir
    os.makedirs(output_dir, exist_ok=True)

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
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Arquivo {file_name} não encontrado em {data_dir}")
            continue
        data = pd.read_csv(file_path)

        # Calcula PF diretamente no DataFrame bruto
        condicoes = [
            data['Rk'] < 7,
            (data['Rk'] >= 7) & (data['Rk'] <= 14),
            data['Rk'] > 14
        ]
        valores = [0, 1, 2]
        data.loc[:, 'PF'] = np.select(condicoes, valores, default=2)

        data_temporal = data[(data['year'] >= 2009) & (data['year'] <= 2019)].copy()
        data_money = data_temporal[(data_temporal['fee_cleaned'].notna()) & (data_temporal['fee_cleaned'] != 0)].copy()

        position_mapping = {
            'GK': 'GK',
            'RB': 'DEF', 'CB': 'DEF', 'LB': 'DEF', 'D': 'DEF',
            'AM': 'MID', 'CM': 'MID', 'DM': 'MID', 'LM': 'MID', 'RM': 'MID',
            'CF': 'STK', 'RW': 'STK', 'LW': 'STK', 'ST': 'STK'
        }
        data_money.loc[:, 'position'] = data_money['position'].map(position_mapping).fillna(data_money['position'])

        data_1_base = gerar_data_1_base(data_money)
        data_2_base = gerar_data_2_base(data_temporal)

        data_1_1year = gerar_dataset_1_resumido(data_1_base, n_anos=1)
        data_1_3years = gerar_dataset_1_resumido(data_1_base, n_anos=3)
        data_1_5years = gerar_dataset_1_resumido(data_1_base, n_anos=5)
        data_2_1year = gerar_dataset_desempenho_temporal(data_2_base, n_anos=1)
        data_2_3years = gerar_dataset_desempenho_temporal(data_2_base, n_anos=3)
        data_2_5years = gerar_dataset_desempenho_temporal(data_2_base, n_anos=5)

        data_3_1year = gerar_dataset_3_combinado(data_1_1year, data_2_1year, n_anos=1)
        data_3_3years = gerar_dataset_3_combinado(data_1_3years, data_2_3years, n_anos=3)
        data_3_5years = gerar_dataset_3_combinado(data_1_5years, data_2_5years, n_anos=5)

        data_1_1year.to_csv(os.path.join(output_dir, f'data_1_1year_{league_name}.csv'), index=False)
        data_1_3years.to_csv(os.path.join(output_dir, f'data_1_3years_{league_name}.csv'), index=False)
        data_1_5years.to_csv(os.path.join(output_dir, f'data_1_5years_{league_name}.csv'), index=False)
        data_2_1year.to_csv(os.path.join(output_dir, f'data_2_1year_{league_name}.csv'), index=False)
        data_2_3years.to_csv(os.path.join(output_dir, f'data_2_3years_{league_name}.csv'), index=False)
        data_2_5years.to_csv(os.path.join(output_dir, f'data_2_5years_{league_name}.csv'), index=False)
        data_3_1year.to_csv(os.path.join(output_dir, f'data_3_1year_{league_name}.csv'), index=False)
        data_3_3years.to_csv(os.path.join(output_dir, f'data_3_3years_{league_name}.csv'), index=False)
        data_3_5years.to_csv(os.path.join(output_dir, f'data_3_5years_{league_name}.csv'), index=False)

        all_data_1_1year.append(data_1_1year)
        all_data_1_3years.append(data_1_3years)
        all_data_1_5years.append(data_1_5years)
        all_data_2_1year.append(data_2_1year)
        all_data_2_3years.append(data_2_3years)
        all_data_2_5years.append(data_2_5years)
        all_data_3_1year.append(data_3_1year)
        all_data_3_3years.append(data_3_3years)
        all_data_3_5years.append(data_3_5years)

    if all_data_1_1year:
        data_1_1year_all = pd.concat(all_data_1_1year, ignore_index=True)
        data_1_3years_all = pd.concat(all_data_1_3years, ignore_index=True)
        data_1_5years_all = pd.concat(all_data_1_5years, ignore_index=True)
        data_2_1year_all = pd.concat(all_data_2_1year, ignore_index=True)
        data_2_3years_all = pd.concat(all_data_2_3years, ignore_index=True)
        data_2_5years_all = pd.concat(all_data_2_5years, ignore_index=True)
        data_3_1year_all = pd.concat(all_data_3_1year, ignore_index=True)
        data_3_3years_all = pd.concat(all_data_3_3years, ignore_index=True)
        data_3_5years_all = pd.concat(all_data_3_5years, ignore_index=True)

        data_1_1year_all.to_csv(os.path.join(output_dir, 'data_1_1year_all_leagues.csv'), index=False)
        data_1_3years_all.to_csv(os.path.join(output_dir, 'data_1_3years_all_leagues.csv'), index=False)
        data_1_5years_all.to_csv(os.path.join(output_dir, 'data_1_5years_all_leagues.csv'), index=False)
        data_2_1year_all.to_csv(os.path.join(output_dir, 'data_2_1year_all_leagues.csv'), index=False)
        data_2_3years_all.to_csv(os.path.join(output_dir, 'data_2_3years_all_leagues.csv'), index=False)
        data_2_5years_all.to_csv(os.path.join(output_dir, 'data_2_5years_all_leagues.csv'), index=False)
        data_3_1year_all.to_csv(os.path.join(output_dir, 'data_3_1year_all_leagues.csv'), index=False)
        data_3_3years_all.to_csv(os.path.join(output_dir, 'data_3_3years_all_leagues.csv'), index=False)
        data_3_5years_all.to_csv(os.path.join(output_dir, 'data_3_5years_all_leagues.csv'), index=False)

        print("Datasets gerados e salvos com sucesso em data/pre_process_repeated!")
    else:
        print("Nenhum dataset foi gerado. Verifique os arquivos de entrada.")

if __name__ == '__main__':
    main()
