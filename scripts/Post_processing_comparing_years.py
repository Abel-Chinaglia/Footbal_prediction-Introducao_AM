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

# 7. Friedman + Nemenyi (Comparando os melhores modelos de cada liga, por duração)
sig = {}
pvals = {}
nemenyi_results = {}

for ds in best['dataset'].unique():
    df_ds = best[best['dataset'] == ds]

    # Criar uma tabela: index = liga, columns = duration, values = melhor accuracy (melhor modelo por liga e duração)
    pivot_data = []

    for duration in x_orders:
        df_duration = df_ds[df_ds['duration'] == duration]

        # Para cada liga, pegar o modelo com maior acurácia naquela duração
        for league in df_duration['league'].unique():
            df_league = df_duration[df_duration['league'] == league]

            if df_league.empty:
                continue

            best_row = df_league.loc[df_league['accuracy'].idxmax()]
            pivot_data.append({
                'league': league,
                'duration': duration,
                'accuracy': best_row['accuracy']
            })

    # Montar o DataFrame final (ligas x durations)
    df_best = pd.DataFrame(pivot_data)
    df_best['accuracy'] = pd.to_numeric(df_best['accuracy'], errors='coerce')
    piv = df_best.pivot(index='league', columns='duration', values='accuracy')

    # Remover ligas que tenham NaN em alguma duração
    piv = piv.dropna()

    # Continuar só se houver pelo menos 2 ligas com dados completos
    if piv.shape[0] < 2:
        print(f"Dataset {ds} - Não há pelo menos 2 ligas com dados completos em todas as durações.")
        sig[ds] = False
        pvals[ds] = np.nan
        continue

    try:
        # Friedman: comparando as durações
        stat = friedmanchisquare(*[piv[col] for col in x_orders])
        pval = stat.pvalue
        pvals[ds] = pval
        signif = pval < 0.05
        sig[ds] = signif

        print(f"Dataset {ds} - p-valor Friedman: {pval:.5f}")
              

        if signif:
            df_long = piv.reset_index().melt(id_vars='league', var_name='duration', value_name='accuracy')
        
            df_long['duration'] = df_long['duration'].astype(str).str.strip()
            df_long['accuracy'] = pd.to_numeric(df_long['accuracy'], errors='coerce')
            df_long = df_long.dropna()
        
            # Transformar em categoria e reindexar
            df_long['league'] = df_long['league'].astype('category')
            df_long['duration'] = df_long['duration'].astype('category')
            df_long['league'] = df_long['league'].cat.set_categories(sorted(df_long['league'].unique()))
            df_long['duration'] = df_long['duration'].cat.set_categories(sorted(df_long['duration'].unique()))
        
            nemenyi = sp.posthoc_nemenyi_friedman(df_long, y_col='accuracy', block_col='league', group_col='duration')
            nemenyi_results[ds] = nemenyi
        
            outpath = os.path.join(datapath, f'nemenyi_{ds}_by_duration_best_per_league.xlsx')
            nemenyi.to_excel(outpath)

    except Exception as e:
        print(f"Erro no Friedman/Nemenyi para dataset {ds}: {e}")
        sig[ds] = False
        pvals[ds] = np.nan


# 8. Plotagem com duas legendas e significância visual
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

    for lg in leagues:
        seq = dfb[dfb['league'] == lg].set_index('duration').reindex(x_orders)
        y = seq['accuracy'].values
        marker = marker_map[lg]

        # Linha tracejada
        ax.plot(x_pos, y, linestyle='--', color='black')

        # Pontos coloridos
        for xi, yi, md in zip(x_pos, y, seq['model']):
            ax.scatter(xi, yi, marker=marker, s=120, facecolor=color_map[md], edgecolor='black')

        # Asteriscos para Friedman
        if sig.get((ds, lg), False):
            ylim = ax.get_ylim()
            y_asterisk = y[0] + (ylim[1] - ylim[0]) * 0.03
            ax.text(x_pos[0], y_asterisk, '*', color='red', fontsize=16, ha='center')

            # Marcação dos pares do Nemenyi
            nemenyi = nemenyi_results[(ds, lg)]
            for i in range(len(x_orders)):
                for j in range(i + 1, len(x_orders)):
                    d1, d2 = x_orders[i], x_orders[j]
                    pval = nemenyi.loc[d1, d2]
                    if pval < 0.05:
                        y_max = max(y[i], y[j]) + (ylim[1] - ylim[0]) * 0.05
                        ax.plot([x_pos[i], x_pos[j]], [y_max, y_max], color='red', linewidth=1.5)
                        ax.text((x_pos[i] + x_pos[j]) / 2, y_max + (ylim[1] - ylim[0]) * 0.015, '*',
                                color='red', fontsize=14, ha='center')

    # Eixos
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title(f"Melhores Modelos - {ds}")
    ax.set_xlabel('Duration')
    ax.set_ylabel('Mean Accuracy')

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


