import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import Machine_learning_boosted_Monteiro as mlb

# Usa caminhos e configurações do código original
input_dir = mlb.input_dir
output_dir = mlb.output_dir
models = mlb.models
param_grids = mlb.param_grids
calcula_metricas = mlb.calcula_metricas

# Garante pastas para .pkl
os.makedirs(output_dir, exist_ok=True)

# 1. Re-executa treinamento, salva melhores estimators como .pkl
for file in os.listdir(input_dir):
    if not file.endswith('.csv'): continue
    data = pd.read_csv(os.path.join(input_dir, file))
    data = data.drop(columns=[col for col in ['league_name','club_name','year'] if col in data.columns])
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    if not np.issubdtype(y.dtype, np.integer):
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)

    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, clf in models.items():
        # Grid search no conjunto completo para obter melhor modelo final
        pipe = Pipeline([('scaler', mlb.StandardScaler()), ('clf', clf)])
        grid = GridSearchCV(pipe, param_grids[model_name], cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1)
        grid.fit(X, y)
        best = grid.best_estimator_
        # Salva o modelo treinado
        pkl_path = os.path.join(output_dir, f"{file[:-4]}_{model_name}.pkl")
        joblib.dump(best, pkl_path)

# 2. Identifica melhores modelos por média de balanced_accuracy
melhores = {}
for file in os.listdir(output_dir):
    if file.endswith('_medias_metricas.xlsx'):
        df = pd.read_excel(os.path.join(output_dir, file))
        nome = file.replace('_medias_metricas.xlsx','')
        melhores[nome] = df.set_index('Modelo')['balanced_accuracy'].idxmax()

# 3. Visualização: frequência de melhores
sns.set(style='whitegrid', palette='pastel', font_scale=1.1)
plt.figure(figsize=(8,4))
sns.countplot(x=list(melhores.values()), order=pd.Series(list(melhores.values())).value_counts().index)
plt.title('Frequência de Melhores Modelos por Dataset')
plt.xlabel('Modelo')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

# 4. Criando ensemble para cada dataset usando hard voting e validando com cv_outer
for nome, best_model in melhores.items():
    print(f"\n[INFO] Processando ensemble para dataset: {nome}")
    
    df = pd.read_excel(os.path.join(output_dir, f"{nome}_medias_metricas.xlsx"))
    top3 = df.sort_values('balanced_accuracy', ascending=False)['Modelo'][:3].tolist()

    csv = os.path.join(input_dir, f"{nome}.csv")
    df_data = pd.read_csv(csv)
    df_data = df_data.drop(columns=[col for col in ['league_name','club_name','year'] if col in df_data.columns])
    X = df_data.iloc[:,:-1].values
    y = df_data.iloc[:,-1].values
    from sklearn.preprocessing import LabelEncoder
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)

    estimators = []
    for tag in top3:
        pkl = os.path.join(output_dir, f"{nome}_{tag}.pkl")
        if os.path.exists(pkl):
            estimators.append((tag, joblib.load(pkl)))
        else:
            print(f"[AVISO] Modelo {pkl} não encontrado. Pulando ensemble para {nome}.")
            estimators = []
            break
    if not estimators: continue

    ensemble = VotingClassifier(estimators=estimators, voting='hard')
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(ensemble, X, y, cv=cv_outer)

    print(f"Ensemble para {nome} com top3: {top3}")
    print(classification_report(y, y_pred))
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap='Blues')
    disp.ax_.set_title(f'Confusion Matrix - Ensemble {nome}')
    plt.show()

    # Salva relatório
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_path = os.path.join(output_dir, f"{nome}_ensemble_report.xlsx")
    report_df.to_excel(report_path)

    # Salva matriz de confusão
    plt.figure()
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap='Blues')
    disp.plot()
    plt.title(f'Matriz de Confusão - Ensemble {nome}')
    plt.savefig(os.path.join(output_dir, f"{nome}_ensemble_confusion_matrix.png"))
    plt.close()

    # Salva predições
    df_preds = pd.DataFrame({'y_true': y, 'y_pred': y_pred})
    df_preds.to_csv(os.path.join(output_dir, f"{nome}_ensemble_preds.csv"), index=False)

    # Treina ensemble completo e salva modelo
    ensemble.fit(X, y)
    joblib.dump(ensemble, os.path.join(output_dir, f"{nome}_ensemble.pkl"))# 5. Salva decisão final dos melhores modelos em Excel
df_best = pd.DataFrame.from_dict(melhores, orient='index', columns=['melhor_modelo']).reset_index()
df_best.rename(columns={'index':'dataset'}, inplace=True)
df_best.to_excel(os.path.join(output_dir,'melhores_modelos_decisao.xlsx'), index=False)
print('Planilha de decisão final salva.')

