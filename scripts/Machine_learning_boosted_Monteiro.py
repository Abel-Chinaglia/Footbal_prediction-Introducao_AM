# Machine_learning_boosted_completo.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
input_dir = os.path.join(parent_dir, 'data', 'pre_process_boost_test')
output_dir = os.path.join(parent_dir, 'results_boost_test')
os.makedirs(output_dir, exist_ok=True)

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
    {'clf__kernel': ['sigmoid'], 'clf__C': [0.1, 1], 'clf__gamma': ['scale', 0.01], 'clf__coef0': [0.0, 0.5]}],
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
    if y_proba is not None and n_classes > 1:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            metrics['roc_auc_macro'] = np.nan
    return metrics

for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        print(f"\nProcessando: {file}")
        data = pd.read_csv(os.path.join(input_dir, file))
        data = data.drop(columns=[col for col in ['league_name', 'club_name', 'year'] if col in data.columns])

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        if not np.issubdtype(y.dtype, np.integer):
            le = LabelEncoder()
            y = le.fit_transform(y)

        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        resultados = {m: [] for m in models}
        y_true_all, y_pred_all = {m: [] for m in models}, {m: [] for m in models}

        for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
            print(f"  Fold externo {fold}/5")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, clf in models.items():
                print(f"    Modelo: {model_name}")
                pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
                param_grid = param_grids[model_name]

                if param_grid:
                    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                else:
                    best_model = pipeline.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test) if hasattr(best_model.named_steps['clf'], "predict_proba") else None

                metrics = calcula_metricas(y_test, y_pred, y_proba, n_classes=len(np.unique(y)))
                resultados[model_name].append(metrics)
                y_true_all[model_name].extend(y_test)
                y_pred_all[model_name].extend(y_pred)

                if y_proba is not None and len(np.unique(y)) > 1:
                    y_test_bin = label_binarize(y_test, classes=np.unique(y))
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                    roc_auc = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f'AUC: {roc_auc:.2f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('FPR')
                    plt.ylabel('TPR')
                    plt.title(f'ROC - {model_name}')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f'{file[:-4]}_roc_{model_name}.png'))
                    plt.close()

        # Médias
        medias_dict = {m: pd.DataFrame(res).mean().to_dict() for m, res in resultados.items()}
        all_metrics_df = pd.concat([pd.DataFrame(m).assign(Modelo=k) for k, m in resultados.items()])
        all_metrics_df.to_excel(os.path.join(output_dir, f'{file[:-4]}_resultados_folds.xlsx'), index=False)

        df_medias = pd.DataFrame(medias_dict).T.reset_index().rename(columns={'index': 'Modelo'})
        df_medias.to_excel(os.path.join(output_dir, f'{file[:-4]}_medias_metricas.xlsx'), index=False)

        # Boxplots
        plt.figure(figsize=(14, 10))
        for i, metrica in enumerate(['balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 1):
            plt.subplot(2, 2, i)
            ax = sns.boxplot(x='Modelo', y=metrica, data=all_metrics_df, palette="Set2")
            medias = all_metrics_df.groupby('Modelo')[metrica].mean()
            for xtick, modelo in enumerate(medias.index):
                ax.text(xtick, medias.loc[modelo] + 0.01, f'{medias.loc[modelo]:.2f}', ha='center', weight='bold')
            plt.title(metrica)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file[:-4]}_boxplot_metricas.png'))
        plt.close()

        # Matriz de confusão
        fig, axes = plt.subplots(2, (len(models) + 1) // 2, figsize=(5*((len(models) + 1)//2), 10))
        axes = axes.flatten()
        for ax, model_name in zip(axes, models):
            cm = confusion_matrix(y_true_all[model_name], y_pred_all[model_name])
            acc = accuracy_score(y_true_all[model_name], y_pred_all[model_name])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}\nAcurácia: {acc:.3f}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Verdadeiro')
        for i in range(len(models), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file[:-4]}_matriz_confusao.png'))
        plt.close()

        # Estatística
        for metrica in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']:
            df_met = all_metrics_df.pivot(columns='Modelo', values=metrica)
        
            # Teste de Friedman
            stat, p = friedmanchisquare(*[df_met[c].dropna() for c in df_met.columns])
        
            # Salva valor de p em um arquivo
            with open(os.path.join(output_dir, f'{file[:-4]}_friedman_{metrica}.txt'), 'w') as f:
                f.write(f'Estatística de Friedman: {stat:.4f}\n')
                f.write(f'p-valor: {p:.6f}\n')
        
            # Teste post hoc apenas se p < 0.05
            if p < 0.05:
                posthoc = sp.posthoc_nemenyi_friedman(df_met.values)
                posthoc.columns = df_met.columns
                posthoc.index = df_met.columns
                posthoc.to_excel(os.path.join(output_dir, f'{file[:-4]}_posthoc_nemenyi_{metrica}.xlsx'))


print("\nProcessamento concluído para todos os arquivos CSV.")
