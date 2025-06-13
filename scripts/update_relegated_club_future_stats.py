# =============================================================================
# update_relegated_club_future_stats.py
# =============================================================================
# Autor: Abel Gonçalves Chinaglia
# Doutorando no PPGRDF - FMRP - USP
# Data: 5 jun. 2025
# Versão do Python: 3.11

# Descrição:
# ----------
# Este script atualiza os dados futuros de desempenho de clubes rebaixados em datasets
# temporais (data_2 e data_3) com informações históricas não padrão do ano imediatamente
# anterior. O objetivo é evitar que valores futuros nulos/padrão prejudiquem a modelagem
# de aprendizado de máquina ao fornecer estimativas mais realistas baseadas em dados prévios.

# Funcionalidades Principais:
# ----------------------------
# - Identifica valores padrão (PF=2, estatísticas zeradas) em colunas de anos futuros
# - Busca dados reais do ano anterior para substituir valores padrão
# - Atualiza arquivos para janelas temporais de 3 e 5 anos (data_2 e data_3)
# - Garante continuidade temporal nos dados para clubes rebaixados
# - Trabalha com arquivos separados por liga e com agregação geral (all_leagues)
# - Realiza cópia segura de arquivos antes da modificação

# Execução:
# ---------
# - Certifique-se de que os arquivos estejam na pasta: ../data/pre_process/
# - Execute o script com:
#   $ python update_relegated_club_future_stats.py
# - Os arquivos corrigidos serão salvos em:
#   ../data/pre_process_repeated/

# Estrutura dos Arquivos Atualizados:
# -----------------------------------
# Arquivos esperados:
#   * data_2_3years_*.csv
#   * data_2_5years_*.csv
#   * data_3_3years_*.csv
#   * data_3_5years_*.csv
# Colunas atualizadas:
#   * {col}_N_years_future para colunas de desempenho (PF, Rk, W, D, L, GF, GA, GD, Pts/MP)
#   * PF_next_N_years (variável alvo da regressão/classificação)

# Licença:
# --------
# Este programa está licenciado sob a GNU Lesser General Public License v3.0.
# Para mais informações, acesse: https://www.gnu.org/licenses/lgpl-3.0.html

# =============================================================================

import os
import pandas as pd
import numpy as np
import shutil

def update_relegated_teams(data, n_anos):
    """
    Atualiza as métricas de desempenho futuras de clubes rebaixados com dados do ano imediatamente anterior não padrão.
    
    Args:
        data (pd.DataFrame): DataFrame contendo os dados do dataset (data_2 ou data_3).
        n_anos (int): Número de anos futuros (3 ou 5).
    
    Returns:
        pd.DataFrame: DataFrame com métricas de desempenho futuras atualizadas.
    """
    valores_padrao = {'PF': 2, 'Rk': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts/MP': 0}
    colunas_desempenho = ['PF', 'Rk', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts/MP']
    
    def get_previous_non_default_data(club, ano_futuro, data, ano_base, n_anos):
        """
        Busca dados do ano imediatamente anterior com valores não padrão.
        
        Args:
            club (str): Nome do clube.
            ano_futuro (int): Ano futuro a ser preenchido.
            data (pd.DataFrame): DataFrame com os dados.
            ano_base (int): Ano base da linha atual.
            n_anos (int): Número de anos futuros (3 ou 5).
        
        Returns:
            pd.Series: Dados do ano anterior mais recente não padrão ou None se não encontrado.
        """
        # Calcula o deslocamento do ano futuro em relação ao ano base
        deslocamento = ano_futuro - ano_base
        
        # Se for o último ano (PF_next_{n_anos}_years), busca o ano imediatamente anterior
        if deslocamento == n_anos:
            previous_deslocamento = deslocamento - 1
            if previous_deslocamento == 0:
                # Busca nas colunas base (ano atual)
                colunas = colunas_desempenho
                grupo_anterior = data[(data['club_name'] == club) & (data['year'] == ano_base)]
                if not grupo_anterior.empty:
                    return grupo_anterior[colunas].iloc[0]
            else:
                # Busca nas colunas do ano futuro anterior (por exemplo, _4_years_future para _5_years)
                colunas = [f'{col}_{previous_deslocamento}_years_future' for col in colunas_desempenho]
                grupo_anterior = data[(data['club_name'] == club) & (data['year'] == ano_base)]
                if not grupo_anterior.empty:
                    valores = grupo_anterior[colunas].iloc[0]
                    # Verifica se os valores não são padrão
                    if not all(valores[f'{col}_{previous_deslocamento}_years_future'] == valores_padrao[col] for col in colunas_desempenho):
                        return valores.rename({f'{col}_{previous_deslocamento}_years_future': col for col in colunas_desempenho})
        
        # Para anos futuros intermediários, tenta o ano imediatamente anterior
        if deslocamento > 1:
            previous_deslocamento = deslocamento - 1
            colunas = [f'{col}_{previous_deslocamento}_years_future' for col in colunas_desempenho]
            grupo_anterior = data[(data['club_name'] == club) & (data['year'] == ano_base)]
            if not grupo_anterior.empty:
                valores = grupo_anterior[colunas].iloc[0]
                # Verifica se os valores não são padrão
                if not all(valores[f'{col}_{previous_deslocamento}_years_future'] == valores_padrao[col] for col in colunas_desempenho):
                    return valores.rename({f'{col}_{previous_deslocamento}_years_future': col for col in colunas_desempenho})
        
        # Se não encontrou dados não padrão, tenta as colunas base
        colunas = colunas_desempenho
        grupo_anterior = data[(data['club_name'] == club) & (data['year'] == ano_base)]
        if not grupo_anterior.empty:
            return grupo_anterior[colunas].iloc[0]
        
        # Se nada for encontrado, tenta anos anteriores ao ano base
        ano_anterior = ano_base - 1
        while ano_anterior >= 2009:
            grupo_anterior = data[(data['club_name'] == club) & (data['year'] == ano_anterior)]
            if not grupo_anterior.empty:
                return grupo_anterior[colunas].iloc[0]
            ano_anterior -= 1
        return None

    # Cria uma cópia do DataFrame para evitar modificações indesejadas
    data_updated = data.copy()
    
    # Identifica as colunas futuras (para 3 ou 5 anos)
    colunas_futuras = []
    for anos_futuros in range(1, n_anos):
        colunas_futuras.extend([f'{col}_{anos_futuros}_years_future' for col in colunas_desempenho])
    colunas_futuras.append(f'PF_next_{n_anos}_years')
    
    # Itera sobre cada linha do DataFrame
    for idx, row in data_updated.iterrows():
        club = row['club_name']
        year = row['year']
        
        # Verifica e atualiza colunas futuras intermediárias (1 a n_anos-1)
        for anos_futuros in range(1, n_anos):
            colunas_ano = [f'{col}_{anos_futuros}_years_future' for col in colunas_desempenho]
            # Verifica se todas as colunas de desempenho do ano futuro têm valores padrão
            is_default = all(
                row[f'{col}_{anos_futuros}_years_future'] == valores_padrao[col]
                for col in colunas_desempenho
            )
            if is_default:
                # Busca dados do ano imediatamente anterior não padrão
                ano_futuro = year + anos_futuros
                previous_data = get_previous_non_default_data(club, ano_futuro, data_updated, year, n_anos)
                if previous_data is not None:
                    for col in colunas_desempenho:
                        data_updated.at[idx, f'{col}_{anos_futuros}_years_future'] = previous_data[col]
        
        # Verifica e atualiza a coluna PF_next_{n_anos}_years
        if row[f'PF_next_{n_anos}_years'] == valores_padrao['PF']:
            ano_alvo = year + n_anos
            previous_data = get_previous_non_default_data(club, ano_alvo, data_updated, year, n_anos)
            if previous_data is not None:
                data_updated.at[idx, f'PF_next_{n_anos}_years'] = previous_data['PF']
    
    return data_updated

def main():
    # Caminhos e configurações
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    pre_process_dir = os.path.join(parent_dir, 'data', 'pre_process')
    output_dir = os.path.join(parent_dir, 'data', 'pre_process_repeated')
    
    # Cria a pasta pre_process_repeated, se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Copia todos os arquivos de pre_process para pre_process_repeated
    for file_name in os.listdir(pre_process_dir):
        src_path = os.path.join(pre_process_dir, file_name)
        dst_path = os.path.join(output_dir, file_name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f'Arquivo copiado: {file_name} -> {dst_path}')
    
    # Arquivos a serem processados
    file_patterns = ['data_2_3years', 'data_2_5years', 'data_3_3years', 'data_3_5years']
    leagues = ['all_leagues', 'bundesliga', 'la_liga', 'ligue_1_fr', 'premier_league', 'serie_a_it']
    
    for pattern in file_patterns:
        n_anos = 3 if '3years' in pattern else 5
        for league in leagues:
            file_name = f'{pattern}_{league}.csv'
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                print(f'Processando {file_name}...')
                # Lê o arquivo
                data = pd.read_csv(file_path)
                # Atualiza os dados de clubes rebaixados
                data_updated = update_relegated_teams(data, n_anos)
                # Salva o arquivo atualizado na pasta pre_process_repeated
                data_updated.to_csv(file_path, index=False)
                print(f'Arquivo atualizado: {file_name} em {output_dir}')
            else:
                print(f'Arquivo {file_name} não encontrado em {output_dir}')

if __name__ == '__main__':
    main()
