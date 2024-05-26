# investlink_data_science_pipeline

Este repositório contém um pipeline completo de ciência de dados para coletar, processar, armazenar, analisar e treinar modelos de machine learning usando os indicadores financeiros das ações do IBXX.

## Estrutura do Projeto

```
IBXX-Data-Pipeline/
│
├── web_scraping/
│   ├── __init__.py
│   ├── scraper.py
│   └── ...
│
├── data_processing/
│   ├── __init__.py
│   ├── cleaner.py
│   └── ...
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py
│   └── ...
│
├── analysis/
│   ├── __init__.py
│   ├── analyzer.py
│   └── ...
│
├── models/
│   ├── __init__.py
│   ├── classifier.py
│   └── regressor.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
│
├── utils/
│   ├── __init__.py
│   └── helpers.py
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Funcionalidades

- **Coleta de Dados**: Scripts para web scraping dos indicadores financeiros das ações do IBXX.
- **Processamento de Dados**: Limpeza e processamento dos dados coletados.
- **Armazenamento de Dados**: Gerenciamento e armazenamento dos dados em um banco de dados.
- **Análise de Dados**: Análise exploratória dos dados.
- **Modelos de Machine Learning**: Treinamento e avaliação de modelos de machine learning para prever se uma ação está cara, barata ou neutra, e prever o valor da ação com base nos indicadores financeiros.
- **Notebooks Jupyter**: Notebooks para exploração e análise interativa dos dados.

## Instalação

### Pré-requisitos

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Passos

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/IBXX-Data-Pipeline.git
   cd IBXX-Data-Pipeline
   ```

2. Construa e inicie os serviços com Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Uso

### Coleta de Dados

Para iniciar a coleta de dados, execute o script `scraper.py`:
```bash
python web_scraping/scraper.py
```

### Processamento de Dados

Para processar os dados coletados, execute o script `cleaner.py`:
```bash
python data_processing/cleaner.py
```

### Análise de Dados

Para realizar a análise dos dados, utilize os notebooks Jupyter disponíveis no diretório `notebooks/`.

### Treinamento de Modelos

Para treinar os modelos de machine learning, execute os scripts no diretório `models/` ou utilize os notebooks Jupyter.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

Para mais informações, entre em contato com Jefferson Valandro em [jeffev123@gmail.com](mailto:jeffev123@gmail.com).
