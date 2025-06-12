import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import joblib
import Machine_learning_boosted_Monteiro as mlb

# Caminhos do módulo mlb
input_dir = mlb.input_dir
output_dir = mlb.output_dir
os.makedirs(output_dir, exist_ok=True)

# Identifica top 3 modelos com maior balanced_accuracy
melhores = {}
for file in os.listdir(output_dir):
    if file.endswith('_medias_metricas.xlsx'):
        df = pd.read_excel(os.path.join(output_dir, file))
        nome = file.replace('_medias_metricas.xlsx', '')
        nome_csv = nome.replace('_ensemble', '') if nome.endswith('_ensemble') else nome
        melhores[nome_csv] = df.sort_values('balanced_accuracy', ascending=False)['Modelo'][:3].tolist()

# Avalia ensemble para cada dataset
for nome_csv, top3 in melhores.items():
    print(f"\n[INFO] Processando ensemble para: {nome_csv} com top3: {top3}")

    csv_path = os.path.join(input_dir, f"{nome_csv}.csv")
    if not os.path.exists(csv_path):
        print(f"[ERRO] Arquivo {csv_path} não encontrado. Pulando...")
        continue

    df_data = pd.read_csv(csv_path)
    df_data = df_data.drop(columns=[col for col in ['league_name', 'club_name', 'year'] if col in df_data.columns])

    X = df_data.iloc[:, :-1].values
    y = df_data.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados, y_true_all, y_pred_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        print(f"  Fold {fold}/5")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimators = []
        skip = False
        for tag in top3:
            model_path = os.path.join(output_dir, f"{nome_csv}_{tag}.pkl")
            if os.path.exists(model_path):
                estimators.append((tag, joblib.load(model_path)))
            else:
                print(f"[ERRO] Modelo {tag} não encontrado em {model_path}")
                skip = True
                break
        if skip: break

        ensemble = VotingClassifier(estimators=estimators, voting='hard')
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        resultados.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
        })

    # Salva resultados dos folds
    df_folds = pd.DataFrame(resultados)
    df_folds['Modelo'] = 'Ensemble_top3'
    df_folds.to_excel(os.path.join(output_dir, f"{nome_csv}_ensemble_resultados_folds.xlsx"), index=False)

    # Salva média das métricas
    df_medias = df_folds.drop(columns='Modelo').mean().to_frame().T
    df_medias['Modelo'] = 'Ensemble_top3'
    df_medias.to_excel(os.path.join(output_dir, f"{nome_csv}_ensemble_medias_metricas.xlsx"), index=False)

    # Boxplots das métricas
    plt.figure(figsize=(14, 10))
    for i, metrica in enumerate(['balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df_folds, y=metrica)
        plt.title(metrica)
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{nome_csv}_ensemble_boxplot_metricas.png"))
    plt.close()

    # Matriz de confusão final
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_true_all, y_pred_all), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Ensemble {nome_csv} - acc: {accuracy_score(y_true_all, y_pred_all):.3f}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{nome_csv}_ensemble_matriz_confusao.png"))
    plt.close()

print("\n[FINALIZADO] Todos os ensembles foram processados e salvos com métricas.")

