**NimbusVita — Dashboard de Saúde e Clima**

Breve descrição

Este projeto é um painel interativo (Dash) para explorar e analisar um conjunto de dados que combina variáveis climáticas, informações demográficas e indicadores de saúde. O objetivo é facilitar a análise exploratória, visualizar padrões e comparar resultados de modelos de classificação e clusterização aplicados ao mesmo conjunto de dados.

Principais recursos

- Aplicação Dash com navegação por abas: **Visão Geral**, **Análise Exploratória**, **Classificação & SHAP**, **Clusterização**, **Prognóstico** e **Pipeline & Tuning** (`dashboard/app_complete.py`).
- Sistema de temas com toggle e persistência (dark/light) (`dashboard/components_theme.py`, `dashboard/core/theme_manager.py`).
- Componentes de UI reutilizáveis: cards com gradiente ou glassmorphism, KPI cards, badges e alertas responsivos (`dashboard/components.py`, `examples_visual_improvements.py`).
- Template visual consistente para Plotly (cores, legendas, fundo), aplicado por `dashboard/core/theme.py::apply_plotly_template`.
- Filtros interativos e estratificação (por `Gênero`, faixas de `Idade`) e alertas automáticos sobre qualidade dos dados (dados insuficientes, classes desbalanceadas).
- Integração completa com modelos de ML:
   - Classificação: treinamento, avaliação e exportação de métricas; comparação entre modelos; suporte a SMOTE.
   - Hyperparameter tuning: GridSearchCV e RandomizedSearchCV com armazenamento dos melhores parâmetros.
   - Explicabilidade: cálculo de SHAP (TreeExplainer) e geração de visualizações (feature importance, beeswarm, force plots) (`src/models/classifier.py`).
- Clusterização:
   - Suporta K-Means e K-Modes (quando a dependência `kmodes` estiver presente).
   - Busca automática de `k` ótimo (silhouette e método do cotovelo) e métricas de avaliação (silhouette, Davies–Bouldin, Calinski–Harabasz).
   - Redução de dimensionalidade via PCA para visualização (PCA 3D).
- DataContext centralizado que carrega o dataset, EDA e tenta carregar modelos pré-treinados de `models/saved_models/` (`dashboard/core/data_context.py`).
- Scripts e utilitários:
   - `scripts/train_models.py`: CLI para treinar classificador e clusterizador (opções: SMOTE, tuning, escolha de algoritmo).
   - `train_interactive.py`: assistente interativo para treinar com diferentes níveis de complexidade e SHAP.
   - `examples_visual_improvements.py`: exemplos prontos de layout, KPIs e templates para reaproveitamento.

Estrutura resumida

```
PISI3-2025.2/
├── data/                       # Datasets (ex.: DATASET FINAL WRDP.csv)
├── dashboard/                  # Aplicação Dash (app_complete.py)
├── scripts/                    # Scripts para treinar modelos e gerar análises
│   └── train_models.py
├── src/                        # Helpers de processamento e modelos
├── requirements.txt
└── README.md
```

Como começar (rápido)

1. Clone o repositório

   git clone https://github.com/Frankiinhu/PISI3-2025.2.git
   cd PISI3-2025.2

2. Crie e ative um ambiente virtual

   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

3. Instale dependências

   pip install -r requirements.txt

Treinar modelos (opcional)

Coloque o arquivo de dados em `data/DATASET FINAL WRDP.csv` e rode:

```
python scripts/train_models.py --data "data/DATASET FINAL WRDP.csv" \
    --out-dir dashboard/models/saved_models \
    --classifier-name classifier_model.pkl \
    --clusterer-name clustering_model.pkl
```

Se preferir já usar modelos pré-treinados, coloque-os em `dashboard/models/saved_models/`.

Executar o dashboard

```
python -m dashboard.app_complete
```

Abra http://127.0.0.1:8050 no navegador.

Formato esperado do dataset

- Variáveis climáticas (ex.: `Temperatura (°C)`, `Umidade`, `Velocidade do Vento (km/h)`).
- Dados demográficos (ex.: `Idade`, `Gênero`).
- Indicadores/sintomas binários (0/1).
- Coluna alvo (ex.: `Diagnóstico`).

Dicas rápidas de solução de problemas

- Modelos não aparecem: confirme que os arquivos `.pkl` estão em `dashboard/models/saved_models/`.
- Erros de dependência: ative o ambiente virtual e reinstale `pip install -r requirements.txt`.
- Colunas do dataset diferentes: atualize o `DataLoader` em `src/data_processing/` ou ajuste os nomes das colunas.

Contribuições e licença

Contribuições são bem-vindas. Este projeto está disponível sob licença MIT.

---
