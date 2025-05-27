# Previsão de Desempenho de Clubes de Futebol com Aprendizado de Máquina

Este projeto aplica algoritmos de aprendizado de máquina supervisionado para prever o desempenho futuro de clubes de futebol das cinco principais ligas europeias (Bundesliga, Premier League, La Liga, Serie A e Ligue 1). Utilizando dados de transferências de jogadores e desempenho em temporadas anteriores, o objetivo é auxiliar os clubes a tomar decisões mais informadas sobre investimentos no mercado de transferências, considerando o impacto financeiro crescente dessas transações.

## Objetivos

### Geral
- Aplicar algoritmos de aprendizado de máquina para predizer o desempenho futuro de times de futebol com base em suas atividades no mercado de transferências e desempenho em temporadas anteriores, abrangendo as temporadas de 2009/2010 a 2019/2020.

### Específicos
- Comparar a capacidade de predição dos algoritmos com diferentes filtros temporais (1, 3 e 5 anos anteriores).
- Comparar a acurácia das predições utilizando diferentes conjuntos de variáveis preditoras:
  1. Dados de transferências de jogadores.
  2. Dados de desempenho nas temporadas anteriores.
  3. Combinação dos dois anteriores.
- Comparar a acurácia das predições entre as diferentes ligas europeias.

## Dados
Os dados utilizados neste projeto foram extraídos de duas fontes principais:
- **Dados de transferências de jogadores**: Disponíveis em [github.com/ewenne/transfers](https://github.com/ewenne/transfers).
- **Dados de classificação dos clubes**: Extraídos manualmente de [fbref.com](https://fbref.com) para conformidade com os termos de uso, cobrindo temporadas de 1992/1993 a 2021/2022.

Os datasets estão organizados no diretório `data/` e incluem arquivos CSV para cada liga:
- `bundesliga_full.csv` (13.446 linhas)
- `la_liga_full.csv` (15.140 linhas)
- `ligue_1_fr_full.csv` (15.764 linhas)
- `premier_league_full.csv` (22.976 linhas)
- `serie_a_it_full.csv` (27.147 linhas)

Cada arquivo contém colunas como nome do clube, nome do jogador, idade, posição, valor da negociação, vitórias, gols, pontos, entre outros.

## Metodologia

### Pré-processamento
- **Filtragem temporal**: Temporadas de 2009/2010 a 2019/2020, excluindo o período da pandemia de COVID-19.
- **Seleção de dados**: Apenas transferências envolvendo dinheiro (excluindo empréstimos gratuitos ou fim de contrato).
- **Bancos de dados**:
  1. **Transferências**: Idade, janela de transferência, valor e quantidade de contratações/vendas por posição (goleiros, defesa, meio-campo, ataque).
  2. **Desempenho anterior**: Posição, vitórias, empates, derrotas, gols feitos, gols sofridos, saldo de gols e pontos por partida.
  3. **Combinado**: Variáveis dos bancos 1 e 2.
- **Variável alvo**: Posição final na temporada seguinte, categorizada em: topo (1º-6º), meio (7º-14º) e base (15º-20º).
- **Padronização**: Uso de `StandardScaler` para normalizar as variáveis.

### Algoritmos
Os seguintes modelos de classificação serão utilizados via Scikit-learn:
- Random Forest
- Logistic Regression
- K-nearest Neighbors
- Support Vector Machine
- Gaussian Naive Bayes

### Avaliação
- **Otimização**: Grid Search para hiperparâmetros, priorizando acurácia balanceada.
- **Validação**: Validação cruzada com k=5.
- **Métricas**: Acurácia, acurácia balanceada, precisão, revocação e F1 Score.

### Análise Estatística
- Comparação das métricas com testes estatísticos (ANOVA ou Kruskal-Wallis, conforme normalidade dos dados), com significância de p<0,05.

## Estrutura do Repositório
- **`data/`**: Datasets em CSV para cada liga.
- **`models/`**: Futuro armazenamento de modelos treinados (atualmente com `deletar_depois.pkl` como placeholder).
- **`scripts/`**: Scripts Python, incluindo `test.py` como ponto de partida.
- **`text/`**: Arquivos LaTeX para documentação (`main.tex`, `Makefile`, `mplainnat.bst`, `refs.bib`).
- **`Proposta_Abel_Rafael_Monteiro-5955006_Introducao_aprendizado_maquina.pdf`**: Proposta completa do projeto.
- **`README.md`**: Este arquivo.

## Como Executar
1. Clone o repositório:
   ```bash
   git clone https://github.com/Abel-Chinaglia/Introducao_AM-5955006.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script principal (quando disponível):
   ```bash
   python scripts/test.py
   ```

**Nota**: Atualize esta seção com instruções específicas conforme os scripts forem desenvolvidos.

## Resultados Esperados
- Desempenho dos algoritmos deve variar entre as ligas, sem indicativo prévio de qual será superior.
- Dados de desempenho anterior devem superar os de transferências isoladamente, mas a combinação de ambos deve apresentar os melhores resultados.
- Filtros temporais mais longos (ex., 5 anos) devem melhorar a acurácia.

## Contribuições
Contribuições são bem-vindas! Siga os passos:
1. Faça um fork do repositório.
2. Crie uma branch (`git checkout -b feature/nova-feature`).
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`).
4. Push para a branch (`git push origin feature/nova-feature`).
5. Abra um Pull Request.

## Disponibilidade
O projeto está versionado no GitHub, e os códigos e dados serão disponibilizados ao final.

## Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
