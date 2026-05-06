# Explicabilidade na Predicao de Dropout em MOOCs com SHAP

## Descricao do Projeto

Este projeto aborda a explicabilidade na predicao de evasao de estudantes em
cursos online massivos abertos (MOOCs), utilizando dados educacionais do
**Open University Learning Analytics Dataset (OULAD)**. A proposta nao e apenas
classificar estudantes como propensos ou nao ao abandono, mas compreender quais
dimensoes comportamentais e academicas orientam as predicoes dos modelos.

O projeto combina dados de perfil do estudante, interacoes no ambiente virtual de
aprendizagem (VLE) e informacoes de avaliacoes para treinar e comparar modelos
classificadores. Em seguida, aplica SHAP (SHapley Additive exPlanations) para
quantificar a importancia das features e comparar tres grupos teoricos:
**perfil**, **interacao** e **avaliacao**.

## Tema e Problema Abordado

MOOCs atendem grandes volumes de estudantes, mas apresentam altas taxas de
abandono. Embora a predicao de dropout seja amplamente estudada, muitos
trabalhos tratam o problema apenas como uma tarefa de classificacao, sem explicar
quais fatores efetivamente orientam as predicoes.

O problema tratado neste projeto e:

> Quais dimensoes de dados - perfil do estudante, interacao com a plataforma ou
> desempenho em avaliacoes - mais contribuem para as predicoes de dropout em
> MOOCs quando analisadas com SHAP?

Esse foco em explicabilidade e importante porque gestores educacionais precisam
entender por que um estudante foi sinalizado como risco de evasao. Assim, o
trabalho conecta resultados de modelos de aprendizado de maquina a dimensoes
pedagogicas interpretaveis.

## Objetivo do Trabalho

O objetivo principal e investigar a explicabilidade das predicoes de dropout em
MOOCs por meio de SHAP, identificando quais grupos de caracteristicas mais
influenciam as decisoes dos modelos.

Objetivos especificos:

- Construir uma base consolidada a partir das tabelas do OULAD.
- Organizar as features em tres grupos teoricos: perfil, interacao e avaliacao.
- Aplicar tratamento de correlacao entre variaveis usando PCA quando necessario.
- Treinar e comparar diferentes algoritmos de classificacao como suporte a
  analise explicativa.
- Calcular a importancia das features por SHAP em cada modelo.
- Agregar os valores SHAP por grupo de features.
- Identificar, por consenso entre modelos, quais dimensoes dominam as predicoes
  de dropout.

## Tecnicas e Modelos Utilizados

O pipeline implementa as seguintes etapas tecnicas:

- **Feature engineering** a partir de dados de perfil, interacoes no VLE e
  avaliacoes.
- **Codificacao de variaveis categoricas** com `LabelEncoder`.
- **Padronizacao de variaveis numericas** com `StandardScaler`.
- **Analise de correlacao** entre features.
- **PCA** para reduzir grupos de variaveis altamente correlacionadas.
- **Selecao greedy por F-score** para escolher features relevantes por grupo.
- **Validacao cruzada** para avaliar estabilidade dos modelos.
- **Avaliacao em hold-out** com separacao por estudantes.
- **Explicabilidade com SHAP** para interpretacao global das features.
- **Agregacao de SHAP por grupos** para comparar perfil, interacao e avaliacao.
- **Consenso entre modelos** para identificar os grupos mais relevantes de forma
  robusta.

Modelos comparados:

- Logistic Regression
- Random Forest
- XGBoost
- Multilayer Perceptron (MLP)

Metricas utilizadas:

- Accuracy
- F1-score para a classe `Withdrawn`
- ROC-AUC

## Bibliotecas e Ferramentas Empregadas

Principais bibliotecas utilizadas:

- `pandas`: manipulacao e consolidacao dos dados.
- `numpy`: operacoes numericas.
- `matplotlib`: geracao de graficos.
- `seaborn`: visualizacao de correlacoes.
- `scikit-learn`: pre-processamento, PCA, selecao de features, validacao,
  metricas e modelos.
- `xgboost`: treinamento do modelo XGBoost.
- `shap`: explicabilidade dos modelos.
- `kagglehub`: download do dataset OULAD.

Ferramentas e artefatos do projeto:

- `notebooks/treinamento algoritmos.py`: pipeline principal de treinamento,
  avaliacao e explicabilidade.
- `notebooks/Coleta_Dataset.ipynb`: notebook relacionado a coleta/preparacao dos
  dados.
- `data/oulad_dropout.csv`: base consolidada utilizada no projeto.
- `results/`: resultados, tabelas e visualizacoes geradas.

## Estrutura do Projeto

```text
predi/
|-- data/
|   |-- oulad_dropout.csv
|   `-- correlacao.png
|-- notebooks/
|   |-- Coleta_Dataset.ipynb
|   `-- treinamento algoritmos.py
|-- results/
|   |-- comparativo_modelos.csv
|   |-- comparativo_metricas.png
|   |-- consenso_grupos.csv
|   |-- features_pca_log.csv
|   |-- shap_features.png
|   |-- shap_features_XGBoost.csv
|   |-- shap_por_grupo.csv
|   `-- shap_por_modelo.png
`-- README.md
```

## Como Executar o Projeto

### 1. Criar e ativar um ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependencias

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### 3. Executar o pipeline principal

Na raiz do projeto, execute:

```bash
python "notebooks/treinamento algoritmos.py"
```

O script baixa o dataset OULAD via `kagglehub`, realiza o processamento dos
dados, treina os modelos, calcula as metricas, executa a analise SHAP e gera os
arquivos de saida.

### 4. Verificar os resultados

Os principais resultados podem ser consultados na pasta `results/`, incluindo:

- `comparativo_modelos.csv`: comparacao das metricas dos modelos.
- `comparativo_metricas.png`: visualizacao comparativa das metricas.
- `shap_por_grupo.csv`: importancia media SHAP por grupo de features.
- `shap_features_XGBoost.csv`: importancia SHAP das features para o melhor
  modelo.
- `consenso_grupos.csv`: consenso dos grupos mais importantes entre os modelos.

## Resultados Obtidos

Nos resultados ja gerados no projeto, o modelo **XGBoost** apresentou o melhor
equilibrio preditivo no conjunto de teste, com ROC-AUC de 91,8% e F1-score de
72,9% para a classe `Withdrawn`.

Na analise SHAP por grupos, os grupos **interacao** e **avaliacao** aparecem como
os mais relevantes para explicar as predicoes de dropout. Cada um domina em dois
dos quatro modelos avaliados, enquanto o grupo **perfil** nao domina em nenhum.
A feature `clicks_2a_metade` apresenta a maior importancia individual no XGBoost,
indicando que o engajamento na segunda metade do curso e um sinal critico para a
explicacao do risco de abandono.
