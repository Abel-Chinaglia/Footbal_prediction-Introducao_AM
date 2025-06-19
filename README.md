# Previsão de Desempenho de Clubes de Futebol com Aprendizado de Máquina

Este projeto aplica algoritmos de aprendizado de máquina supervisionado para prever o desempenho futuro de clubes de futebol das cinco principais ligas europeias (Bundesliga, Premier League, La Liga, Serie A e Ligue 1). Utilizando dados de transferências de jogadores e desempenho em temporadas anteriores, o objetivo é fornecer suporte estratégico baseado em dados para decisões no mercado de transferências.

## Objetivos

### Geral
Aplicar algoritmos de aprendizado de máquina para predizer o desempenho futuro de clubes de futebol com base em atividades no mercado de transferências e desempenho histórico nas temporadas anteriores, no intervalo de 2009/2010 a 2019/2020.

### Específicos
- Avaliar o impacto de diferentes filtros temporais (1, 3 e 5 anos) nas variáveis preditoras.
- Comparar o desempenho dos modelos com diferentes conjuntos de variáveis:
  1. Somente transferências.
  2. Somente desempenho histórico.
  3. Combinação de ambos.
- Avaliar os resultados separadamente para cada liga.
- Analisar o impacto de diferentes estratégias de tratamento para clubes rebaixados.

## Conjunto de Dados

Os dados foram extraídos de:
- [github.com/ewenme/transfers](https://github.com/ewenme/transfers) – transferências de jogadores.
- [fbref.com](https://fbref.com) – desempenho esportivo dos clubes (extração manual para obedecer aos termos de uso).

As variáveis incluem:
- Informações de transferências: valor, posição, tipo, idade dos jogadores, entre outras.
- Métricas de desempenho: vitórias, empates, derrotas, gols, pontos, saldo de gols, etc.
- Estatísticas avançadas (xG, clean sheets, artilheiro).

## Pré-processamento

- Foco no período de 2009/2010 a 2019/2020, evitando vieses da pandemia.
- Exclusão de transferências sem compensação financeira (empréstimos gratuitos ou fim de contrato).
- Construção de três bancos:
  1. Transferências.
  2. Desempenho anterior.
  3. Combinação dos dois.

### Estratégias para Clubes Rebaixados
- **Exclusão**: remoção completa quando faltam dados futuros.
- **Preenchimento com zeros**: uso de valores padrão para manter integridade.
- **Repetição**: replicação dos últimos dados disponíveis.

Cada estratégia gerou um subconjunto de dados salvo em diretórios separados.

## Modelagem

Modelos utilizados:
- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes
- AdaBoost
- XGBoost

### Validação e Otimização
- Validação cruzada (5-fold interna e externa).
- Busca de hiperparâmetros via Grid Search.
- Padronização dos dados via `StandardScaler`.

### Métricas de Avaliação
- Acurácia
- Acurácia Balanceada
- Precisão
- Revocação
- F1-Score
- AUC-ROC (quando aplicável)

### Ensemble
Os três modelos com melhor F1-Score foram combinados em um ensemble (VotingClassifier) com estratégia de votação `soft` (quando possível) ou `hard`.

## Análise Estatística

- Boxplots com métricas de desempenho.
- Matrizes de confusão.
- Testes de Friedman e post-hoc de Nemenyi (`scikit-posthocs`) para avaliar diferenças significativas entre modelos.

## Estrutura do Projeto

```
.
├── data/
│   ├── pre_process/
│   ├── pre_process_boost/
│   ├── pre_process_repeated/
│   └── pre_process_without_relegated/
├── models/
├── results/
├── results_boost/
├── results_repeated_boost/
├── results_repeated_boost_ensemble/
├── results_without_relegated/
├── scripts/
│   └── __pycache__/
├── text/
├── environment.yml
├── requirements.txt
├── README.md
└── Apresentacao_IAM-Abel_Rafael_2025.pptx
```

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/Abel-Chinaglia/Footbal_prediction-Introducao_AM.git
   cd Footbal_prediction-Introducao_AM
   ```

2. Crie o ambiente:
   ```bash
   conda env create -f environment.yml
   conda activate futebol_ml_env
   ```

   Ou instale via pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os scripts desejados no diretório `scripts/`.


## Contribuições
Contribuições são bem-vindas! Siga os passos:
1. Faça um fork do repositório.
2. Crie uma branch (`git checkout -b feature/nova-feature`).
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`).
4. Push para a branch (`git push origin feature/nova-feature`).
5. Abra um Pull Request.

## Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
