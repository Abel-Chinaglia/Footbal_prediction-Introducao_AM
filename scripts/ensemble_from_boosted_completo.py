import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import joblib
import Machine_learning_boosted_Monteiro as mlb

input_dir = mlb.input_dir
output_dir = mlb.output_dir
models = mlb.models
param_grids = mlb.param_grids
calcula_metricas = mlb.calcula_metricas
os.makedirs(output_dir, exist_ok=True)

# === 1. Treina e salva modelos base
for file in os.listdir(input_dir):
    if not file.endswith('.csv'): continue
    data = pd.read_csv(os.path.join(input_dir, file))
    data = data.drop(columns=[col for col in ['league_name','club_name','year'] if col in data.columns])
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)

    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, clf in models.items():
        pipe = Pipeline([('scaler', mlb.StandardScaler()), ('clf', clf)])
        grid = GridSearchCV(pipe, param_grids[model_name], cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
        grid.fit(X, y)
        best = grid.best_estimator_
        pkl_path = os.path.join(output_dir, f"{file[:-4]}_{model_name}.pkl")
        joblib.dump(best, pkl_path)

# === 2. Identifica melhores modelos por balanced_accuracy (top1)
melhores = {}
for file in os.listdir(output_dir):
    if file.endswith('_medias_metricas.xlsx'):
        df = pd.read_excel(os.path.join(output_dir, file))
        nome = file.replace('_medias_metricas.xlsx','')
        melhores[nome] = df.set_index('Modelo')['balanced_accuracy'].idxmax()

# === 3. Visualiza frequência dos melhores modelos (opcional)
sns.set(style='whitegrid', palette='pastel', font_scale=1.1)
plt.figure(figsize=(8,4))
sns.countplot(x=list(melhores.values()), order=pd.Series(list(melhores.values())).value_counts().index)
plt.title('Frequência de Melhores Modelos por Dataset')
plt.xlabel('Modelo')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()

# === 4. Avalia ensemble com os top 3 modelos usando CV externa e salva resultados detalhados
# Identifica top3 para cada dataset
top3_models = {}
for file in os.listdir(output_dir):
    if file.endswith('_medias_metricas.xlsx'):
        df = pd.read_excel(os.path.join(output_dir, file))
        nome = file.replace('_medias_metricas.xlsx','')
        top3_models[nome] = df.sort_values('balanced_accuracy', ascending=False)['Modelo'][:3].tolist()

for nome, top3 in top3_models.items():
    print(f"\n[INFO] Ensemble para: {nome} com top3 modelos: {top3}")

    csv = os.path.join(input_dir, f"{nome}.csv")
    df_data = pd.read_csv(csv)
    df_data = df_data.drop(columns=[col for col in ['league_name','club_name','year'] if col in df_data.columns])
    X = df_data.iloc[:, :-1].values
    y = df_data.iloc[:, -1].values
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados, y_true_all, y_pred_all = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y), 1):
        print(f"  Fold externo {fold}/5")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimators = []
        skip = False
        for tag in top3:
            model_path = os.path.join(output_dir, f"{nome}_{tag}.pkl")
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

    # Salva resultados por fold
    df_folds = pd.DataFrame(resultados)
    df_folds['Modelo'] = 'Ensemble_top3'
    df_folds.to_excel(os.path.join(output_dir, f"{nome}_ensemble_resultados_folds.xlsx"), index=False)

    # Salva médias das métricas
    df_medias = df_folds.drop(columns='Modelo').mean().to_frame().T
    df_medias['Modelo'] = 'Ensemble_top3'
    df_medias.to_excel(os.path.join(output_dir, f"{nome}_ensemble_medias_metricas.xlsx"), index=False)

    # Boxplots das métricas
    plt.figure(figsize=(14, 10))
    for i, metrica in enumerate(['balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df_folds, y=metrica)
        plt.title(metrica)
        plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{nome}_ensemble_boxplot_metricas.png"))
    plt.close()

    # Matriz de confusão consolidada
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Ensemble {nome} - acc: {accuracy_score(y_true_all, y_pred_all):.3f}')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{nome}_ensemble_matriz_confusao.png"))
    plt.close()

    # Opcional: salva relatório detalhado
    report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_excel(os.path.join(output_dir, f"{nome}_ensemble_classification_report.xlsx"))

    # Treina ensemble final em todos os dados e salva modelo
    final_estimators = []
    for tag in top3:
        pkl = os.path.join(output_dir, f"{nome}_{tag}.pkl")
        final_estimators.append((tag, joblib.load(pkl)))
    final_ensemble = VotingClassifier(estimators=final_estimators, voting='hard')
    final_ensemble.fit(X, y)
    joblib.dump(final_ensemble, os.path.join(output_dir, f"{nome}_ensemble.pkl"))

print("\n[FINALIZADO] Todos os ensembles processados e salvos com métricas.")

