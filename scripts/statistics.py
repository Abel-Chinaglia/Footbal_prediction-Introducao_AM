# =============================================================================
# statistics.py
# =============================================================================
# Autor: Rafael Luiz Martins Monteiro
# Doutorando no PPGRDF - FMRP - USP
# Data: 13 jun. 2025
# Versão do Python: 3.11

# Descrição:
# ----------
# Este script realiza análises estatísticas comparativas de resultados de modelos de aprendizado
# de máquina, focando em comparações temporais, entre ligas e entre datasets. Ele lê arquivos
# Excel contendo resultados de validação cruzada, seleciona os melhores modelos por acurácia
# média e aplica testes estatísticos de Friedman e pós-teste de Nemenyi para avaliar diferenças
# significativas no desempenho.

# Funcionalidades Principais:
# ----------------------------
# - Lê arquivos Excel de resultados de validação cruzada de um diretório especificado
# - Calcula a acurácia média e seleciona o melhor modelo por dataset, duração e liga
# - Filtra os resultados dos melhores modelos para análises estatísticas
# - Realiza três tipos de comparações:
#   1. Temporal: compara diferentes durações (1, 3, 5 anos) dentro de cada dataset e liga
#   2. Entre ligas: compara ligas dentro de cada dataset e duração
#   3. Entre datasets: compara datasets dentro de cada liga e duração
# - Aplica o teste de Friedman e o pós-teste de Nemenyi para cada comparação
# - Salva resultados detalhados (estatísticas, p-valores) e resumos em arquivos CSV
# - Organiza os resultados em subdiretórios específicos para cada tipo de comparação

# Execução:
# ---------
# - Certifique-se de que os arquivos Excel de entrada estejam no diretório: D:/Introducao_AM-5955006/results_repeated_boost_ensemble
# - Execute o script com:
#   $ python statistics.py
# - O script processará os arquivos Excel e salvará os resultados em subdiretórios:
#   - comparacoes_temporais/
#   - comparacoes_ligas/
#   - comparacoes_datasets/

# Formato dos Dados de Entrada:
# -----------------------------
# Arquivos Excel contendo colunas com informações de modelo, acurácia e outros metadados
# (dataset, duração, liga). Os nomes dos arquivos devem seguir o padrão:
# '*_resultados_folds.xlsx', onde partes do nome indicam dataset, duração e liga.

# Estrutura de Saída:
# -------------------
# - Subdiretórios com resultados organizados por tipo de comparação:
#   - comparacoes_temporais: resultados de testes temporais (1, 3, 5 anos)
#   - comparacoes_ligas: resultados de comparações entre ligas
#   - comparacoes_datasets: resultados de comparações entre datasets
# - Arquivos CSV por comparação, contendo:
#   - Estatísticas de Friedman (estatística e p-valor)
#   - Resultados do pós-teste de Nemenyi
#   - Resumos com médias, desvios padrão e p-valores relevantes

# Dependências:
# -------------
# - pandas, numpy, scipy, scikit-posthocs
# - Instale as dependências com: pip install pandas numpy scipy scikit-posthocs

# Licença:
# --------
# Este programa está licenciado sob a GNU Lesser General Public License v3.0.
# Para mais detalhes, acesse: https://www.gnu.org/licenses/lgpl-3.0.html

# ============================================================================

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# --- 1. Leitura e preparação dos dados --------------------------------------
datapath = 'D:/Introducao_AM-5955006/results_repeated_boost_ensemble'
files   = glob.glob(os.path.join(datapath, '*_resultados_folds.xlsx'))

records = []
for f in files:
    df = pd.read_excel(f)
    parts    = os.path.basename(f).split('_')
    ds       = f"data_{parts[1]}"
    duration = parts[2]
    league   = '_'.join(parts[3:-2])
    df['dataset']  = ds
    df['duration'] = duration
    df['league']   = league
    records.append(df)

data_all = pd.concat(records, ignore_index=True)

# --- 2. Detecta colunas de modelo e accuracy --------------------------------
model_col = next(c for c in data_all.columns if c.lower() in ['modelo','model'])
acc_col   = next(c for c in data_all.columns if 'accuracy' in c.lower())

# --- 3. Encontra melhor modelo por média em cada (dataset, duration, league)---
data_mean = (
    data_all
    .groupby(['dataset','duration','league',model_col])[acc_col]
    .mean()
    .reset_index()
    .rename(columns={model_col:'model', acc_col:'accuracy'})
)

best_models = (
    data_mean
    .loc[data_mean.groupby(['dataset','duration','league'])['accuracy']
         .idxmax()]
    .reset_index(drop=True)
)

# --- 4. Filtra todos os folds dos melhores modelos --------------------------
filtered = pd.merge(
    data_all,
    best_models[['dataset','duration','league','model']],
    how='inner',
    left_on = ['dataset','duration','league', model_col],
    right_on= ['dataset','duration','league','model']
)

# --- 5. Padroniza duração e cria índice de execução 'run'-------------------
filtered['duration'] = filtered['duration'].replace({'3years':'3year','5years':'5year'})
filtered['run']      = filtered.groupby(['dataset','league','duration']).cumcount()

# --- 6. Configura pastas de saída -------------------------------------------
out_temp     = os.path.join(datapath, 'comparacoes_temporais')
out_league   = os.path.join(datapath, 'comparacoes_ligas')
out_dataset  = os.path.join(datapath, 'comparacoes_datasets')
os.makedirs(out_temp,    exist_ok=True)
os.makedirs(out_league,  exist_ok=True)
os.makedirs(out_dataset, exist_ok=True)

# --- 7. Inicializa listas de resumo -----------------------------------------
summary_temp   = []
summary_league = []
summary_dataset = []

# --- 8. Loop 1: comparação temporal dentro de cada league -------------------
for dataset in filtered['dataset'].unique():
    for league in filtered['league'].unique():
        sub = filtered[(filtered['dataset']==dataset)&(filtered['league']==league)]
        if sub.empty: continue

        descr = sub.groupby('duration')[acc_col].agg(['mean','std'])
        if not {'1year','3year','5year'}.issubset(descr.index): continue

        piv = sub.pivot(index='run', columns='duration', values=acc_col).dropna()
        stat,p = friedmanchisquare(piv['1year'],piv['3year'],piv['5year'])
        pd.DataFrame({'statistic':[stat],'p_value':[p]})\
          .to_csv(os.path.join(out_temp, f"friedman_{dataset}_{league}.csv"), index=False)

        post = sp.posthoc_nemenyi_friedman(piv.values)
        post.index = post.columns = ['1year','3year','5year']
        post.to_csv(os.path.join(out_temp, f"posthoc_nemenyi_{dataset}_{league}.csv"))

        summary_temp.append({
            'dataset':dataset,'league':league,
            'mean_1y':descr.loc['1year','mean'],'std_1y':descr.loc['1year','std'],
            'mean_3y':descr.loc['3year','mean'],'std_3y':descr.loc['3year','std'],
            'mean_5y':descr.loc['5year','mean'],'std_5y':descr.loc['5year','std'],
            'fried_stat':stat,'fried_p':p,
            'p_1v3':post.loc['1year','3year'],'p_1v5':post.loc['1year','5year'],'p_3v5':post.loc['3year','5year']
        })
        print(f"[Temporal] {dataset} | {league} concluído.")

# --- 9. Loop 2: comparação entre ligas dentro de cada duration -------------
for dataset in filtered['dataset'].unique():
    for duration in ['1year','3year','5year']:
        sub = filtered[(filtered['dataset']==dataset)&(filtered['duration']==duration)]
        if sub.empty: continue

        descr = sub.groupby('league')[acc_col].agg(['mean','std'])
        leagues = descr.index.tolist()
        if len(leagues) < 2: continue

        piv = sub.pivot(index='run', columns='league', values=acc_col).dropna(axis=0,how='any')
        args = [piv[lg] for lg in leagues]
        stat,p = friedmanchisquare(*args)
        pd.DataFrame({'statistic':[stat],'p_value':[p]})\
          .to_csv(os.path.join(out_league, f"friedman_{dataset}_{duration}.csv"), index=False)

        post = sp.posthoc_nemenyi_friedman(piv.values)
        post.index = post.columns = leagues
        post.to_csv(os.path.join(out_league, f"posthoc_nemenyi_{dataset}_{duration}.csv"))

        entry = {'dataset':dataset,'duration':duration,'fried_stat':stat,'fried_p':p}
        for lg in leagues:
            entry[f"{lg}_mean"] = descr.loc[lg,'mean']
            entry[f"{lg}_std"]  = descr.loc[lg,'std']
        # post-hoc p-values
        for i,lg1 in enumerate(leagues):
            for lg2 in leagues[i+1:]:
                entry[f"p_{lg1}_vs_{lg2}"] = post.loc[lg1,lg2]
        summary_league.append(entry)
        print(f"[Ligas] {dataset} | {duration} concluído.")

# --- 10. Loop 3: comparação entre datasets dentro de cada league e ano -------
for league in filtered['league'].unique():
    for duration in ['1year','3year','5year']:
        sub = filtered[(filtered['league']==league)&(filtered['duration']==duration)]
        if sub.empty: continue

        descr = sub.groupby('dataset')[acc_col].agg(['mean','std'])
        datasets = descr.index.tolist()
        if len(datasets) < 2: continue

        piv = sub.pivot(index='run', columns='dataset', values=acc_col).dropna(axis=0,how='any')
        args = [piv[ds] for ds in datasets]
        stat,p = friedmanchisquare(*args)
        pd.DataFrame({'statistic':[stat],'p_value':[p]})\
          .to_csv(os.path.join(out_dataset, f"friedman_{league}_{duration}.csv"), index=False)

        post = sp.posthoc_nemenyi_friedman(piv.values)
        post.index = post.columns = datasets
        post.to_csv(os.path.join(out_dataset, f"posthoc_nemenyi_{league}_{duration}.csv"))

        entry = {'league':league,'duration':duration,'fried_stat':stat,'fried_p':p}
        for ds in datasets:
            entry[f"{ds}_mean"] = descr.loc[ds,'mean']
            entry[f"{ds}_std"]  = descr.loc[ds,'std']
        # post-hoc p-values datasets
        for i,ds1 in enumerate(datasets):
            for ds2 in datasets[i+1:]:
                entry[f"p_{ds1}_vs_{ds2}"] = post.loc[ds1,ds2]
        summary_dataset.append(entry)
        print(f"[Datasets] {league} | {duration} concluído.")

# --- 11. Salva resumos finais -----------------------------------------------
pd.DataFrame(summary_temp).to_csv(os.path.join(out_temp,   'resumo_temporal.csv'), index=False)
pd.DataFrame(summary_league).to_csv(os.path.join(out_league, 'resumo_ligas.csv'),   index=False)
pd.DataFrame(summary_dataset).to_csv(os.path.join(out_dataset,'resumo_datasets.csv'), index=False)

print("Todas as análises concluídas:\n"
      f"- Temporais: {out_temp}\n"
      f"- Ligas:     {out_league}\n"
      f"- Datasets:  {out_dataset}")



