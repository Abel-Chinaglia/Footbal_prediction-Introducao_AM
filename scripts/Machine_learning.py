import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# === Diretórios ===
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
input_dir = os.path.join(parent_dir, 'data', 'pre_process')
output_dir = os.path.join(parent_dir, 'results')

# Criar pasta de resultados, se não existir
os.makedirs(output_dir, exist_ok=True)

# === Modelos e hiperparâmetros ===
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=False, random_state=42),
    'GaussianNB': GaussianNB()
}

param_grids = {
    'RandomForest': {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5]
    },
    'LogisticRegression': {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs']
    },
    'KNN': {
        'clf__n_neighbors': [3, 5, 7, 9],
        'clf__weights': ['uniform', 'distance']
    },
    'SVM': {
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf']
    },
    'GaussianNB': {}  # Sem hiperparâmetros
}

# === Função de métricas ===
def calcula_metricas(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

# === Loop sobre os arquivos CSV ===
for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        print(f"\n========================\nProcessando arquivo: {file}\n========================")
        
        file_path = os.path.join(input_dir, file)
        data = pd.read_csv(file_path)

        # --- Preparação dos dados ---
        # Remover colunas não usadas
        data = data.drop(columns=[col for col in ['league_name', 'club_name', 'year'] if col in data.columns])

        # Definir X e y
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # === Nested CV ===
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        resultados = {model_name: [] for model_name in models.keys()}
        y_true_all = {model_name: [] for model_name in models.keys()}
        y_pred_all = {model_name: [] for model_name in models.keys()}

        for fold_outer, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
            print(f"\nFold externo {fold_outer}/5")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, clf in models.items():
                print(f"  Modelo: {model_name}")

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', clf)
                ])

                param_grid = param_grids[model_name]
                if param_grid:
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring='balanced_accuracy',
                        cv=cv_inner,
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    print(f"    Melhores hiperparâmetros: {grid_search.best_params_}")
                else:
                    best_model = pipeline.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)
                metricas = calcula_metricas(y_test, y_pred)
                resultados[model_name].append(metricas)

                y_true_all[model_name].extend(y_test)
                y_pred_all[model_name].extend(y_pred)

        # === Resultados médios ===
        medias_dict = {}
        for model_name, metrics_list in resultados.items():
            df_metrics = pd.DataFrame(metrics_list)
            medias = df_metrics.mean().to_dict()
            medias_dict[model_name] = medias

        # === Matrizes de confusão ===
        num_models = len(models)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))

        if num_models == 1:
            axes = [axes]  # Garantir iterabilidade

        for ax, model_name in zip(axes, models.keys()):
            cm = confusion_matrix(y_true_all[model_name], y_pred_all[model_name])
            acc = accuracy_score(y_true_all[model_name], y_pred_all[model_name])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}\nAcurácia: {acc:.3f}')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Verdadeiro')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file[:-4]}_matriz_confusao.png'))
        plt.close()

        # === Boxplots ===
        all_metrics_df = pd.DataFrame()
        for model_name, metrics_list in resultados.items():
            df_temp = pd.DataFrame(metrics_list)
            df_temp['Modelo'] = model_name
            all_metrics_df = pd.concat([all_metrics_df, df_temp], axis=0)

        metricas_para_plot = ['balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        plt.figure(figsize=(14, 10))
        for i, metrica in enumerate(metricas_para_plot, 1):
            plt.subplot(2, 2, i)
            ax = sns.boxplot(x='Modelo', y=metrica, data=all_metrics_df, palette="Set2")

            medias = all_metrics_df.groupby('Modelo')[metrica].mean()
            for xtick, modelo in enumerate(medias.index):
                media_valor = medias.loc[modelo]
                ax.text(xtick, media_valor + 0.01, f'{media_valor:.2f}', 
                        horizontalalignment='center', color='black', weight='bold', fontsize=9)

            plt.title(f'Comparação da métrica: {metrica}')
            plt.xlabel('')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file[:-4]}_boxplot_metricas.png'))
        plt.close()

        # === Salvar resultados ===
        # Resultados de cada fold
        all_metrics_df.to_excel(os.path.join(output_dir, f'{file[:-4]}_resultados_folds.xlsx'), index=False)

        # Médias
        df_medias = pd.DataFrame(medias_dict).T.reset_index().rename(columns={'index': 'Modelo'})
        df_medias.to_excel(os.path.join(output_dir, f'{file[:-4]}_medias_metricas.xlsx'), index=False)

        print(f"\n✅ Resultados salvos para {file} na pasta 'results'.")

print("\n=== Processo concluído para todos os arquivos CSV ===")


