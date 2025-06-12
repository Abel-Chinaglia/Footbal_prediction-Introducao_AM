import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp

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
dir_atual = os.getcwd()
dir_pai = os.path.dirname(dir_atual)
dir_entrada = os.path.join(dir_pai, 'data', 'pre_process_repeated_ensemble')
dir_saida = os.path.join(dir_pai, 'results_repeated_boost_ensemble_test')
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

# Cross-validation externa e interna
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
        le = LabelEncoder()
        y = le.fit_transform(y)

    resultados = {m: [] for m in models}
    y_true_all = {m: [] for m in models}
    y_pred_all = {m: [] for m in models}

    # Avaliação individual dos modelos
    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        print(f"  Fold externo {fold}/5")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for nome, clf in models.items():
            print(f"    Modelo: {nome}")
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            grid = param_grids[nome]
            if grid:
                gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
                gs.fit(X_tr, y_tr)
                best = gs.best_estimator_
            else:
                best = pipe.fit(X_tr, y_tr)

            y_pr = best.predict(X_te)
            y_pb = best.predict_proba(X_te) if hasattr(best.named_steps['clf'], 'predict_proba') else None
            met = calcula_metricas(y_te, y_pr, y_pb, n_classes=len(np.unique(y)))
            resultados[nome].append(met)
            y_true_all[nome].extend(y_te)
            y_pred_all[nome].extend(y_pr)

            # ROC plot
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

    # ------------------ Ensemble ------------------
    # Seleciona top-3 pelo f1_macro médio
    df_medias = pd.DataFrame({m: pd.DataFrame(res).mean() for m, res in resultados.items()}).T
    top_n = 3
    top_models = df_medias.sort_values(by='f1_macro', ascending=False).head(top_n).index.tolist()
    print(f"\nTreinando ensemble com os {top_n} melhores modelos: {top_models}")

    # Recria estimators com melhores hiperparâmetros
    best_ests = []
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
        best_ests.append((nome, best_pipe.named_steps['clf']))

    soft = all(hasattr(est, 'predict_proba') for _, est in best_ests)
    voting = VotingClassifier(estimators=best_ests, voting='soft' if soft else 'hard')

    ens_metrics, y_t_ens, y_p_ens = [], [], []
    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipe_ens = Pipeline([('scaler', StandardScaler()), ('clf', voting)])
        pipe_ens.fit(X_tr, y_tr)
        y_pr = pipe_ens.predict(X_te)
        y_pb = pipe_ens.predict_proba(X_te) if soft else None
        met = calcula_metricas(y_te, y_pr, y_pb, n_classes=len(np.unique(y)))
        ens_metrics.append(met)
        y_t_ens.extend(y_te)
        y_p_ens.extend(y_pr)
    resultados['Ensemble'] = ens_metrics
    y_true_all['Ensemble'] = y_t_ens
    y_pred_all['Ensemble'] = y_p_ens
    # -----------------------------------------------

    # Cálculo de médias e salvamento de resultados
    all_df = pd.concat([pd.DataFrame(res).assign(Modelo=m) for m, res in resultados.items()])
    all_df.to_excel(os.path.join(dir_saida, f'{arquivo[:-4]}_resultados_folds.xlsx'), index=False)
    df_medias = all_df.groupby('Modelo').mean().reset_index()
    df_medias.to_excel(os.path.join(dir_saida, f'{arquivo[:-4]}_medias_metricas.xlsx'), index=False)

    # Gráficos de boxplot e matriz de confusão
    plt.figure(figsize=(14, 10))
    for i, metrica in enumerate(['balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='Modelo', y=metrica, data=all_df, palette="Set2")
        medias = all_df.groupby('Modelo')[metrica].mean()
        for xt, mod in enumerate(medias.index):
            plt.text(xt, medias.loc[mod] + 0.01, f'{medias.loc[mod]:.2f}', ha='center', weight='bold')
        plt.title(metrica)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_saida, f'{arquivo[:-4]}_boxplot_metricas.png'))
    plt.close()

    fig, axes = plt.subplots(2, (len(models)+1)//2, figsize=(5*((len(models)+1)//2), 10))
    axes = axes.flatten()
    for ax, nome in zip(axes, resultados.keys()):
        cm = confusion_matrix(y_true_all[nome], y_pred_all[nome])
        acc = accuracy_score(y_true_all[nome], y_pred_all[nome])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{nome}\nAcurácia: {acc:.3f}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')
    for i in range(len(resultados), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_saida, f'{arquivo[:-4]}_matriz_confusao.png'))
    plt.close()

    # Testes estatísticos
    for metrica in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']:
        piv = all_df.pivot(columns='Modelo', values=metrica)
        stat, p = friedmanchisquare(*[piv[c].dropna() for c in piv.columns])
        with open(os.path.join(dir_saida, f'{arquivo[:-4]}_friedman_{metrica}.txt'), 'w') as f:
            f.write(f'Estatística de Friedman: {stat:.4f}\n')
            f.write(f'p-valor: {p:.6f}\n')
        if p < 0.05:
            post = sp.posthoc_nemenyi_friedman(piv.values)
            post.index = post.columns = piv.columns
            post.to_excel(os.path.join(dir_saida, f'{arquivo[:-4]}_posthoc_nemenyi_{metrica}.xlsx'))

print("\nProcessamento concluído para todos os arquivos CSV.")
