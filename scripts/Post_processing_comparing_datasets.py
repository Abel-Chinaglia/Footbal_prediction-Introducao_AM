import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import matplotlib.lines as mlines

try:
    import scikit_posthocs as sp
except ImportError:
    raise ImportError("Instale com `pip install scikit-posthocs`.")

# 1. Leitura dos arquivos
datapath = 'D:/Introducao_AM-5955006/results_repeated_boost_ensemble'
output_path = os.path.join(datapath, 'comparison_by_dataset')
os.makedirs(output_path, exist_ok=True)

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

# 4. Seleção do melhor modelo por duração e dataset
idx = data_mean.groupby(['dataset', 'duration', 'league'])['accuracy'].idxmax()
best = data_mean.loc[idx].reset_index(drop=True)

# 5. Exporta novo CSV sem sobrescrever o antigo
best.to_csv(os.path.join(output_path, 'final_best_models_by_dataset.csv'), index=False)

# 6. Parâmetros de datasets
dataset_orders = ['data_1', 'data_2', 'data_3']
x_labels = dataset_orders
x_pos = np.arange(len(dataset_orders))

# 7. Friedman + Nemenyi por duração e liga (comparando datasets)
sig = {}
pvals = {}
nemenyi_results = {}

for duration in best['duration'].unique():
    df_dur = data_mean[data_mean['duration'] == duration]
    for lg in df_dur['league'].unique():
        df_lg = df_dur[df_dur['league'] == lg]
        piv = df_lg.pivot(index='model', columns='dataset', values='accuracy')
        try:
            piv = piv.loc[:, dataset_orders]
        except KeyError:
            sig[(duration, lg)] = False
            pvals[(duration, lg)] = np.nan
            continue

        if piv.shape[0] < 2:
            sig[(duration, lg)] = False
            pvals[(duration, lg)] = np.nan
            continue

        try:
            stat = friedmanchisquare(*[piv[d] for d in dataset_orders])
            pval = stat.pvalue
            pvals[(duration, lg)] = pval
            signif = pval < 0.05
            sig[(duration, lg)] = signif

            if signif:
                df_long = piv.reset_index().melt(id_vars='model', var_name='dataset', value_name='accuracy')
                nemenyi = sp.posthoc_nemenyi_friedman(df_long, y_col='accuracy', block_col='model', group_col='dataset')
                nemenyi_results[(duration, lg)] = nemenyi

                # Salva Nemenyi em Excel na nova pasta
                outpath = os.path.join(output_path, f'nemenyi_by_dataset_{duration}_{lg}.xlsx')
                nemenyi.to_excel(outpath)
        except Exception as e:
            print(f"Erro no teste para {duration} - {lg}: {e}")
            sig[(duration, lg)] = False
            pvals[(duration, lg)] = np.nan

# 8. Plotagem: Um gráfico por duração, X = datasets
markers = ['o', 's', 'D', '^', 'v', '<', '>']
models = best['model'].unique()
leagues_all = best['league'].unique()
cmap = plt.cm.tab10
color_map = {m: cmap(i) for i, m in enumerate(models)}
marker_map = {lg: markers[i % len(markers)] for i, lg in enumerate(leagues_all)}

for duration in best['duration'].unique():
    dfb = best[best['duration'] == duration]
    leagues = dfb['league'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    for lg in leagues:
        seq = dfb[dfb['league'] == lg].set_index('dataset').reindex(dataset_orders)
        y = seq['accuracy'].values
        marker = marker_map[lg]

        # Linha tracejada
        ax.plot(x_pos, y, linestyle='--', color='black')

        # Pontos coloridos
        for xi, yi, md in zip(x_pos, y, seq['model']):
            ax.scatter(xi, yi, marker=marker, s=120, facecolor=color_map[md], edgecolor='black')

        # Asterisco se houve significância no Friedman
        if sig.get((duration, lg), False):
            ylim = ax.get_ylim()
            y_asterisk = y[0] + (ylim[1] - ylim[0]) * 0.03
            ax.text(x_pos[0], y_asterisk, '*', color='red', fontsize=16, ha='center')

            # Marcação dos pares significativos do Nemenyi
            nemenyi = nemenyi_results.get((duration, lg))
            if nemenyi is not None:
                for i in range(len(dataset_orders)):
                    for j in range(i + 1, len(dataset_orders)):
                        d1, d2 = dataset_orders[i], dataset_orders[j]
                        pval = nemenyi.loc[d1, d2]
                        if pval < 0.05:
                            y_max = max(y[i], y[j]) + (ylim[1] - ylim[0]) * 0.05
                            ax.plot([x_pos[i], x_pos[j]], [y_max, y_max], color='red', linewidth=1.5)
                            ax.text((x_pos[i] + x_pos[j]) / 2, y_max + (ylim[1] - ylim[0]) * 0.015, '*',
                                    color='red', fontsize=14, ha='center')

    # Eixos
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_title(f"Melhores Modelos por Dataset - {duration}")
    ax.set_xlabel('Dataset')
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
