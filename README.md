# InvestLink — Data Science Pipeline

Pipeline completo de ciência de dados para coletar, processar, treinar modelos de machine learning e gerar previsões de valuation das ações do IBXX, integrado ao banco de dados do InvestLink.

## Estrutura

```
data_science_pipeline/
│
├── web_scraping/                  # Sprint 1 — Coleta de dados
│   ├── scraper_indicators.py      # Selenium + StatusInvest → indicadores financeiros
│   ├── scraper_prices.py          # yfinance → histórico de preços
│   └── run_scraping.py            # Entry point do scraping com CLI
│
├── data_processing/               # Sprint 2 — Feature engineering
│   ├── processor.py               # Limpeza: winsorize, fill nulos por mediana de setor
│   ├── feature_engineer.py        # Z-scores setoriais + scores compostos
│   ├── labeler.py                 # Labels relativos ao Ibovespa (BARATA/NEUTRA/CARA)
│   └── build_training_dataset.py  # Orquestra processamento → data/training_dataset.parquet
│
├── database/                      # Sprint 1 — Persistência
│   ├── connector.py               # Engine SQLAlchemy via DATABASE_URL
│   ├── models.py                  # Tabelas: stock_indicators_history, stock_prices_history, stock_predictions
│   ├── migrations.py              # Cria/atualiza as tabelas no banco
│   └── queries.py                 # Queries reutilizáveis
│
├── models/                        # Sprint 3 — Machine learning
│   ├── trainer.py                 # GradientBoosting vs XGBoost (GridSearchCV + StratifiedKFold 5-fold)
│   ├── evaluator.py               # Métricas cross-validadas + plots
│   └── predictor.py               # Normaliza dados atuais e salva previsões no DB
│
├── analysis/                      # Análises ad-hoc
│
├── notebooks/
│   └── 01_eda.ipynb               # Análise exploratória dos dados
│
├── data/
│   └── training_dataset.parquet   # Gerado por build_training_dataset.py (gitignored)
│
├── pipeline.py                    # Entry point master — orquestra todos os estágios
├── requirements.txt
├── .env.example
└── .gitignore
```

## Lógica do pipeline

```
StatusInvest (Selenium)           yfinance
        │                             │
        ▼                             ▼
stock_indicators_history      stock_prices_history
        │                             │
        └──────────────┬──────────────┘
                       ▼
              processor.py (limpeza)
              feature_engineer.py (z-scores, scores)
              labeler.py (labels vs Ibovespa)
                       │
                       ▼
          data/training_dataset.parquet
                       │
                       ▼
           trainer.py → best_model.joblib
           evaluator.py → métricas
           predictor.py → stock_predictions (DB)
```

### Labels

Alpha = retorno da ação − retorno do Ibovespa no período:

| Label   | Condição         |
|---------|------------------|
| BARATA  | alpha > +15%     |
| CARA    | alpha < −15%     |
| NEUTRA  | entre −15% e +15%|

### Scores compostos (0–100)

| Score    | Peso |
|----------|------|
| value    | 30%  |
| quality  | 35%  |
| growth   | 20%  |
| dividend | 15%  |

## Instalação

### Pré-requisitos

- Python 3.10+
- PostgreSQL (ou via Docker Compose do InvestLink)
- Google Chrome (para o Selenium)

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/jeffev/investlink_data_science_pipeline.git
cd investlink_data_science_pipeline

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure o banco de dados
cp .env.example .env
# Edite .env com a DATABASE_URL correta

# 4. Crie as tabelas
python database/migrations.py
```

### Variáveis de ambiente

```bash
# .env
DATABASE_URL=postgresql://postgres:senha@localhost:5433/investlink
```

A porta padrão em desenvolvimento local é `5433` (mapeada no `docker-compose.yml` do InvestLink).

## Uso

### Pipeline completo (todos os estágios)

```bash
python pipeline.py --all
```

### Estágios individuais

```bash
# Coleta de dados (indicadores + preços)
python pipeline.py --scrape

# Gerar dataset de treinamento
python pipeline.py --dataset

# Treinar modelo
python pipeline.py --train

# Avaliar modelo existente
python pipeline.py --evaluate

# Gerar previsões e salvar no banco
python pipeline.py --predict
```

### Combinações comuns

```bash
# Re-treinar e re-prever (dados já no banco)
python pipeline.py --train --predict

# Scraping de tickers específicos e previsão
python pipeline.py --scrape --predict --tickers VALE3 PETR4 ITUB4

# Pipeline completo com browser visível (debug)
python pipeline.py --all --no-headless

# Previsão simulada sem gravar no banco
python pipeline.py --predict --dry-run
```

### Scraping standalone

```bash
# Todos os indicadores e preços
python web_scraping/run_scraping.py --mode all

# Só indicadores
python web_scraping/run_scraping.py --mode indicators

# Tickers específicos, forçando re-scrape
python web_scraping/run_scraping.py --tickers VALE3 PETR4 --force

# Com browser visível
python web_scraping/run_scraping.py --no-headless
```

### Dataset standalone

```bash
python data_processing/build_training_dataset.py

# Sem labels relativos ao Ibovespa
python data_processing/build_training_dataset.py --no-relative
```

## Dependências principais

| Categoria        | Biblioteca                              |
|------------------|-----------------------------------------|
| Banco de dados   | SQLAlchemy 2.0, psycopg2-binary         |
| Web scraping     | Selenium 4, BeautifulSoup4, yfinance    |
| Processamento    | pandas, numpy                           |
| Machine learning | scikit-learn, XGBoost, joblib           |
| Visualização     | matplotlib, seaborn, jupyter            |
| NLP (Sprint 4)   | transformers, torch                     |
