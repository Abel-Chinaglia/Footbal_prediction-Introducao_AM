# =============================================================================
# machine_learning_boosted_enssemble_v1_0.py
# =============================================================================
# Autor: Rafael Luiz Martins Monteiro
# Doutorando no PPGRDF - FMRP - USP
# Data: 8 jun. 2025
# Versão do Python: 3.11

# Descrição:
# ----------
# Este script realiza tarefas de classificação em conjuntos de dados relacionados à marcha,
# utilizando diversos algoritmos de aprendizado de máquina com validação cruzada aninhada.
# Inclui ajuste de hiperparâmetros via GridSearchCV e gera previsões por meio de um modelo
# de ensemble baseado em votação (soft ou hard). Calcula diversas métricas de desempenho e
# executa testes estatísticos para comparação entre modelos.

# Funcionalidades Principais:
# ----------------------------
# - Implementa diversos classificadores (RandomForest, LogisticRegression, KNN, SVM, etc.)
# - Utiliza validação cruzada aninhada para avaliação robusta dos modelos
# - Realiza otimização de hiperparâmetros com GridSearchCV
# - Calcula métricas como: acurácia, acurácia balanceada, precisão, recall, F1-score e AUC-ROC
# - Constrói modelos de ensemble com os classificadores de melhor desempenho
# - Gera gráficos: curvas ROC, boxplots e matrizes de confusão
# - Realiza testes estatísticos de Friedman e pós-teste de Nemenyi para comparação de desempenho
# - Salva todos os resultados e visualizações em diretório de saída

# Execução:
# ---------
# - Certifique-se de que os arquivos CSV de entrada estejam no diretório: ../data/pre_process_repeated
# - Execute o script com:
#   $ python machine_learning_boosted_enssemble_v1_0.py
# - O script processará cada arquivo CSV, aplicará os modelos e salvará os resultados em:
#   ../results_repeated_boost_ensemble/

# Formato dos Dados de Entrada:
# -----------------------------
# Arquivos CSV contendo características numéricas e a variável alvo na última coluna.
# Colunas opcionais como 'league_name', 'club_name' e 'year' são removidas automaticamente se existirem.

# Estrutura de Saída:
# -------------------
# Para cada arquivo CSV processado:
# - Arquivos Excel com métricas por divisão (fold) e médias
# - Gráficos de curva ROC para modelos com saída de probabilidade
# - Boxplots das principais métricas entre os modelos
# - Matrizes de confusão para cada classificador
# - Resultados dos testes estatísticos de comparação de desempenho

# Licença:
# --------
# Este programa está licenciado sob a GNU Lesser General Public License v3.0.
# Para mais detalhes, acesse: https://www.gnu.org/licenses/lgpl-3.0.html

# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import time
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
)
from xgboost import XGBClassifier
from scipy.stats import friedmanchisquare

# Diretórios
idir_atual = os.getcwd()
dir_pai = os.path.dirname(idir_atual)
dir_entrada = os.path.join(dir_pai, 'data', 'pre_process_repeated')
dir_saida = os.path.join(dir_pai, 'results_repeated_boost_ensemble')
os.makedirs(dir_saida, exist_ok=True)

# Modelos e hiperparâmetros
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'GaussianNB': GaussianNB(),
    'AdaBoost_DT': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
}

param_grids = {
    'RandomForest': {'clf__n_estimators': [50, 100, 1000], 'clf__max_depth': [None, 5], 'clf__min_samples_split': [2, 5]},
    'LogisticRegression': {'clf__C': [0.01, 0.1, 1, 10], 'clf__penalty': ['l2'], 'clf__solver': ['lbfgs']},
    'KNN': {'clf__n_neighbors': [3, 5, 7, 9], 'clf__weights': ['uniform', 'distance']},
    'SVM': [
        {'clf__kernel': ['linear'], 'clf__C': [0.01, 0.1, 1, 10]},
        {'clf__kernel': ['rbf'], 'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto', 0.01, 0.1]},
        {'clf__kernel': ['poly'], 'clf__C': [0.1, 1], 'clf__degree': [2, 3], 'clf__gamma': ['scale', 0.01], 'clf__coef0': [0.0, 0.5]},
        {'clf__kernel': ['sigmoid'], 'clf__C': [0.1, 1], 'clf__gamma': ['scale', 0.01], 'clf__coef0': [0.0, 0.5]}
    ],
    'GaussianNB': {},
    'AdaBoost_DT': {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.1, 1.0]},
    'XGBoost': {'clf__n_estimators': [10, 100, 500], 'clf__max_depth': [3, 5], 'clf__learning_rate': [0.01, 0.1]},
}

def calcula_metricas(y_true, y_pred, y_proba=None, n_classes=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    if y_proba is not None and n_classes and n_classes > 1:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            metrics['roc_auc_macro'] = np.nan
    return metrics

# Estruturas de cross-validation
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for arquivo in os.listdir(dir_entrada):
    if not arquivo.endswith('.csv'):
        continue
    print(f"\nProcessando: {arquivo}")
    df = pd.read_csv(os.path.join(dir_entrada, arquivo))
    df = df.drop(columns=[c for c in ['league_name', 'club_name', 'year'] if c in df.columns])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)

    resultados = {m: [] for m in models}
    y_true_all = {m: [] for m in models}
    y_pred_all = {m: [] for m in models}

    # Avaliação individual com tqdm e postfix de tempo
    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        print(f"  Fold {fold_idx}/{cv_outer.get_n_splits()}")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        with tqdm(models.items(), desc=f'Fold {fold_idx} Modelos', leave=True) as pbar:
            for nome, clf in pbar:
                start = time.time()
                pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
                grid = param_grids[nome]
                if grid:
                    gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
                    gs.fit(X_tr, y_tr)
                    best = gs.best_estimator_
                else:
                    best = pipe.fit(X_tr, y_tr)
                elapsed = time.time() - start
                tqdm.write(f"    {nome} treinado em {elapsed:.2f}s")

                y_pr = best.predict(X_te)
                y_pb = best.predict_proba(X_te) if hasattr(best.named_steps['clf'], 'predict_proba') else None
                met = calcula_metricas(y_te, y_pr, y_pb, n_classes=len(np.unique(y)))
                resultados[nome].append(met)
                y_true_all[nome].extend(y_te)
                y_pred_all[nome].extend(y_pr)

                # ROC
                if y_pb is not None and len(np.unique(y)) > 1:
                    y_te_bin = label_binarize(y_te, classes=np.unique(y))
                    fpr, tpr, _ = roc_curve(y_te_bin.ravel(), y_pb.ravel())
                    roc_auc = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f'AUC: {roc_auc:.2f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('FPR')
                    plt.ylabel('TPR')
                    plt.title(f'ROC - {nome}')
                    plt.legend()
                    plt.savefig(os.path.join(dir_saida, f'{arquivo[:-4]}_roc_{nome}.png'))
                    plt.close()

    # Construção e avaliação do ensemble
    df_medias = pd.DataFrame({m: pd.DataFrame(res).mean() for m, res in resultados.items()}).T
    top_n = 3
    top_models = df_medias.sort_values(by='f1_macro', ascending=False).head(top_n).index.tolist()
    print(f"\nTreinando ensemble com: {top_models}")

    estimators = []
    for nome in top_models:
        clf = models[nome]
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        grid = param_grids[nome]
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
            gs.fit(X, y)
            best_pipe = gs.best_estimator_
        else:
            best_pipe = pipe.fit(X, y)
        estimators.append((nome, best_pipe.named_steps['clf']))
    voting = VotingClassifier(estimators=estimators, voting='soft' if all(hasattr(est, 'predict_proba') for _, est in estimators) else 'hard')

    ens_metrics, y_te_ens, y_pr_ens = [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipe_ens = Pipeline([('scaler', StandardScaler()), ('clf', voting)])
        pipe_ens.fit(X_tr, y_tr)
        y_pr = pipe_ens.predict(X_te)
        y_pb = pipe_ens.predict_proba(X_te) if hasattr(voting, 'predict_proba') else None
        met = calcula_metricas(y_te, y_pr, y_pb, n_classes=len(np.unique(y)))
        ens_metrics.append(met)
        y_te_ens.extend(y_te)
        y_pr_ens.extend(y_pr)
    resultados['Ensemble'] = ens_metrics
    y_true_all['Ensemble'] = y_te_ens
    y_pred_all['Ensemble'] = y_pr_ens

    # Salvamento e estatísticas finais
    all_df = pd.concat([pd.DataFrame(res).assign(Modelo=m) for m, res in resultados.items()])
    all_df.to_excel(os.path.join(dir_saida, f'{arquivo[:-4]}_resultados_folds.xlsx'), index=False)
    df_medias = all_df.groupby('Modelo').mean().reset_index()
    df_medias.to_excel(os.path.join(dir_saida, f'{arquivo[:-4]}_medias_metricas.xlsx'), index=False)

    # Boxplots e matrizes
    plt.figure(figsize=(14, 10))
    for i, metr in enumerate(['balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='Modelo', y=metr, data=all_df, palette='Set2')
        means = all_df.groupby('Modelo')[metr].mean()
        for xt, mdl in enumerate(means.index):
            plt.text(xt, means.loc[mdl] + 0.01, f'{means.loc[mdl]:.2f}', ha='center', weight='bold')
        plt.title(metr)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_saida, f'{arquivo[:-4]}_boxplot_metricas.png'))
    plt.close()

    fig, axes = plt.subplots(2, (len(models)+1)//2, figsize=(5*((len(models)+1)//2), 10))
    axes = axes.flatten()
    for ax, mdl in zip(axes, resultados.keys()):
        cm = confusion_matrix(y_true_all[mdl], y_pred_all[mdl])
        acc = accuracy_score(y_true_all[mdl], y_pred_all[mdl])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{mdl}\nAcurácia: {acc:.3f}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')
    for i in range(len(resultados), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_saida, f'{arquivo[:-4]}_matriz_confusao.png'))
    plt.close()

    # Testes estatísticos
    for metr in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']:
        pivot = all_df.pivot(columns='Modelo', values=metr)
        stat, p = friedmanchisquare(*[pivot[c].dropna() for c in pivot.columns])
        with open(os.path.join(dir_saida, f'{arquivo[:-4]}_friedman_{metr}.txt'), 'w') as f:
            f.write(f'Estatística de Friedman: {stat:.4f}\n')
            f.write(f'p-valor: {p:.6f}\n')
        if p < 0.05:
            posthoc = sp.posthoc_nemenyi_friedman(pivot.values)
            posthoc.index = posthoc.columns = pivot.columns
            posthoc.to_excel(os.path.join(dir_saida, f'{arquivo[:-4]}_posthoc_nemenyi_{metr}.xlsx'))

print("\nProcessamento concluído para todos os arquivos CSV.")

