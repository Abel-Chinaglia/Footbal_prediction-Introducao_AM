# =============================================================================
# post_processing_comparing_years.py
# =============================================================================
# Autor: Rafael Luiz Martins Monteiro
# Doutorando no PPGRDF - FMRP - USP
# Data: 13 jun. 2025
# Versão do Python: 3.11

# Descrição:
# ----------
# Este script realiza o pós-processamento de resultados de modelos de aprendizado de máquina,
# comparando o desempenho de diferentes algoritmos em múltiplos conjuntos de dados com diferentes
# cortes temporais (1 ano, 3 anos, 5 anos). Ele lê arquivos Excel contendo resultados de validação
# cruzada, calcula a acurácia média por modelo, identifica o melhor modelo para cada combinação de
# dataset, duração e liga, e gera visualizações gráficas para comparação de desempenho.

# Funcionalidades Principais:
# ----------------------------
# - Lê arquivos Excel de resultados de validação cruzada de um diretório especificado
# - Calcula a acurácia média por modelo, dataset, duração e liga
# - Seleciona o melhor modelo com base na acurácia média para cada combinação
# - Exporta os resultados dos melhores modelos para um arquivo CSV
# - Gera gráficos comparativos por dataset, com cortes temporais no eixo X e acurácia no eixo Y
# - Utiliza diferentes cores para modelos e marcadores para ligas, com legendas duplas
# - Aplica o teste estatístico de Friedman para comparação de desempenho entre modelos

# Execução:
# ---------
# - Certifique-se de que os arquivos Excel de entrada estejam no diretório: D:/Introducao_AM-5955006/results_repeated_boost_ensemble
# - Execute o script com:
#   $ python post_processing_comparing_years.py
# - O script processará os arquivos Excel e salvará os resultados e gráficos no mesmo diretório

# Formato dos Dados de Entrada:
# -----------------------------
# Arquivos Excel contendo colunas com informações de modelo, acurácia e outros metadados
# (dataset, duração, liga). Os nomes dos arquivos devem seguir o padrão:
# '*_resultados_folds.xlsx', onde partes do nome do arquivo indicam dataset, duração e liga.

# Estrutura de Saída:
# -------------------
# - Arquivo CSV 'final_best_models.csv' com os melhores modelos por dataset, duração e liga
# - Gráficos de linhas tracejadas e pontos coloridos mostrando a acurácia média dos melhores modelos
# - Legendas duplas indicando modelos (por cor) e ligas (por forma do marcador)

# Dependências:
# -------------
# - pandas, numpy, matplotlib, scipy, scikit-posthocs
# - Instale as dependências com: pip install pandas numpy matplotlib scipy scikit-posthocs

# Licença:
# --------
# Este programa está licenciado sob a GNU Lesser General Public License v3.0.
# Para mais detalhes, acesse: https://www.gnu.org/licenses/lgpl-3.0.html

# ============================================================================

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import matplotlib.lines as mlines
import scikit_posthocs as sp


# 1. Leitura dos arquivos
datapath = 'D:/Introducao_AM-5955006/results_repeated_boost_ensemble'
files = glob.glob(os.path.join(datapath, '*_resultados_folds.xlsx'))
records = []
for f in files:
    df = pd.read_excel(f)
    parts = os.path.basename(f).split('_')
    ds = f"data_{parts[1]}"
    duration = parts[2]
    league = '_'.join(parts[3:-2])
    df['dataset'] = ds
    df['duration'] = duration
    df['league'] = league
    records.append(df)
data_all = pd.concat(records, ignore_index=True)

# 2. Identificação das colunas
model_col = next(c for c in data_all.columns if c.lower() in ['modelo', 'model'])
acc_col = next(c for c in data_all.columns if 'accuracy' in c.lower())

# 3. Média de resultados
data_mean = (
    data_all
    .groupby(['dataset', 'duration', 'league', model_col])[acc_col]
    .mean()
    .reset_index()
    .rename(columns={model_col: 'model', acc_col: 'accuracy'})
)

# 4. Seleção do melhor modelo
idx = data_mean.groupby(['dataset', 'duration', 'league'])['accuracy'].idxmax()
best = data_mean.loc[idx].reset_index(drop=True)

# 5. Exporta CSV
best.to_csv(os.path.join(datapath, 'final_best_models.csv'), index=False)

# 6. Parâmetros de duração
x_orders = ['1year', '3years', '5years']
duration_map = {'1year': '1 year', '3years': '3 years', '5years': '5 years'}
x_labels = [duration_map[d] for d in x_orders]
x_pos = np.arange(len(x_orders))


# 7. Plotagem com duas legendas e significância visual
markers = ['o', 's', 'D', '^', 'v', '<', '>']
models = best['model'].unique()
leagues_all = best['league'].unique()
cmap = plt.cm.tab10
color_map = {m: cmap(i) for i, m in enumerate(models)}
marker_map = {lg: markers[i % len(markers)] for i, lg in enumerate(leagues_all)}

for ds in best['dataset'].unique():
    dfb = best[best['dataset'] == ds]
    leagues = dfb['league'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove as bordas superior e esquerda
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    for lg in leagues:
        seq = dfb[dfb['league'] == lg].set_index('duration').reindex(x_orders)
        y = seq['accuracy'].values
        marker = marker_map[lg]

        # Linha tracejada
        ax.plot(x_pos, y, linestyle='--', color='black')

        # Pontos coloridos
        for xi, yi, md in zip(x_pos, y, seq['model']):
            ax.scatter(xi, yi, marker=marker, s=120, facecolor=color_map[md], edgecolor='black')


    # Eixos
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title(f"Melhores Modelos - {ds}")
    ax.set_xlabel('Cortes temporais')
    ax.set_ylabel('Acurácia média (%)')

    # Legenda 1: Modelos (cores)
    model_handles = [
        mlines.Line2D([], [], marker='o', color=color_map[m], linestyle='', markersize=8, label=m)
        for m in models
    ]
    legend1 = ax.legend(handles=model_handles, title='Modelo (cor)', loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend1)

    # Legenda 2: Ligas (formas)
    league_handles = [
        mlines.Line2D([], [], marker=marker_map[lg], color='black', linestyle='--', markersize=8, label=lg)
        for lg in leagues
    ]
    legend2 = ax.legend(handles=league_handles, title='Liga (forma)', loc='upper left', bbox_to_anchor=(1.05, 0.55))
    ax.add_artist(legend2)

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.show()


