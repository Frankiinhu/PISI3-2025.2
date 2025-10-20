# 🌡️ VitaNimbus - Weather Related Disease Prediction

## 📋 Descrição do Projeto

VitaNimbus é um projeto completo de **Análise Exploratória de Dados (EDA)** e **Machine Learning** para predição de doenças relacionadas ao clima. O sistema utiliza técnicas de **Classificação** e **Clusterização** para diagnosticar doenças com base em sintomas e fatores climáticos, além de identificar fatores de risco.

### 🎯 Objetivos

- ✅ Análise Exploratória completa do dataset "Weather Related Disease Prediction"
- ✅ Modelo de Classificação para diagnóstico de doenças
- ✅ Modelo de Clusterização para identificação de padrões e grupos de risco
- ✅ Dashboard interativo em Dash para visualização de dados
- ✅ API REST para integração com aplicativos externos
- ✅ Identificação de fatores de risco climáticos e sintomáticos

---

## 📁 Estrutura do Projeto

```
PISI3-2025.2/
│
├── data/                          # Datasets
│   └── DATASET FINAL WRDP.csv    # Dataset principal
│
├── src/                           # Código fonte
│   ├── data_processing/           # Processamento de dados
│   │   ├── data_loader.py        # Carregamento e limpeza
│   │   └── eda.py                # Análise exploratória
│   │
│   ├── models/                    # Modelos de ML
│   │   ├── classifier.py         # Classificação (diagnóstico)
│   │   └── clustering.py         # Clusterização (padrões)
│   │
│   ├── visualization/             # Visualizações
│   │
│   └── api/                       # API REST
│       └── app.py                # Servidor Flask
│
├── dashboard/                     # Dashboard Dash
│   ├── app.py                    # Aplicação principal
│   ├── components/               # Componentes do dashboard
│   └── assets/                   # Arquivos estáticos (CSS, imagens)
│
├── models/                        # Modelos treinados
│   └── saved_models/             # Modelos salvos (.pkl)
│
├── notebooks/                     # Jupyter Notebooks
│   └── analise_exploratoria.ipynb # Análise inicial
│
├── scripts/                       # Scripts utilitários
│   └── train_models.py           # Treinar modelos
│
├── docs/                         # Documentação
│
├── requirements.txt              # Dependências Python
├── .gitignore                   # Arquivos ignorados pelo Git
└── README.md                    # Este arquivo
```

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)

### 1. Clonar o Repositório

```bash
git clone https://github.com/Frankiinhu/PISI3-2025.2.git
cd PISI3-2025.2
```

### 2. Criar Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## 📊 Uso do Projeto

### 1. Treinar os Modelos

Antes de usar o dashboard ou a API, treine os modelos de Machine Learning:

```bash
cd scripts
python train_models.py
```

Este script irá:
- Carregar e limpar os dados
- Treinar o modelo de classificação (Random Forest)
- Treinar o modelo de clusterização (K-Means)
- Salvar os modelos em `models/saved_models/`

### 2. Executar o Dashboard

```bash
cd dashboard
python app.py
```

Acesse o dashboard em: `http://localhost:8050`

#### Funcionalidades do Dashboard:
- 📊 **Visão Geral**: Estatísticas e distribuições básicas
- 🔍 **Análise Exploratória**: Análise detalhada de sintomas e correlações
- 🌡️ **Clima vs Diagnóstico**: Relações entre variáveis climáticas e doenças
- 💊 **Sintomas**: Heatmaps e análises de sintomas
- 🤖 **Modelos ML**: Métricas, importância de features, clusters
- 🎯 **Predição**: Interface para predição de diagnósticos

### 3. Executar a API

```bash
cd src/api
python app.py
```

A API estará disponível em: `http://localhost:5000`

#### Endpoints Disponíveis:

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Informações da API |
| `/health` | GET | Status da API |
| `/predict` | POST | Predição de diagnóstico |
| `/predict_batch` | POST | Predição em lote |
| `/cluster` | POST | Identificação de cluster |
| `/risk_factors` | POST | Análise de fatores de risco |
| `/model_info` | GET | Informações dos modelos |

#### Exemplo de Uso da API:

```python
import requests
import json

url = "http://localhost:5000/predict"

data = {
    "idade": 35,
    "temperatura": 38.5,
    "umidade": 75,
    "velocidade_vento": 15,
    "sintomas": {
        "Febre": 1,
        "Tosse": 1,
        "Fadiga": 1,
        "Dor de Cabeça": 1,
        "Dor no Peito": 0
    }
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

---

## 🔬 Modelos de Machine Learning

### 1. Classificador (Diagnóstico)

- **Algoritmo**: Random Forest Classifier
- **Features**: Sintomas + Variáveis climáticas + Dados demográficos
- **Target**: Diagnóstico da doença
- **Métricas**: Acurácia, Precision, Recall, F1-Score
- **Validação**: Cross-validation com 5 folds

### 2. Clusterizador (Padrões e Risco)

- **Algoritmo**: K-Means
- **Features**: Sintomas + Variáveis climáticas
- **Objetivo**: Identificar grupos de risco e padrões
- **Métricas**: Silhouette Score, Davies-Bouldin, Calinski-Harabasz

---

## 📈 Análise Exploratória de Dados

### Variáveis Analisadas:

#### Climáticas:
- 🌡️ Temperatura (°C)
- 💧 Umidade (%)
- 💨 Velocidade do Vento (km/h)

#### Demográficas:
- 👤 Idade
- ⚧ Gênero

#### Sintomas:
- Febre, Tosse, Fadiga, Dor de Cabeça
- Náusea, Dor no Peito, Vômito
- E mais de 30 outros sintomas

### Visualizações Incluídas:
- Distribuições univariadas e bivariadas
- Matrizes de correlação
- Boxplots de variáveis climáticas por diagnóstico
- Heatmaps de sintomas por diagnóstico
- Gráficos de importância de features
- Visualização de clusters (PCA)

---

## 🔧 Desenvolvimento e Contribuição

### Adicionar Novas Features

1. **Processamento de Dados**: Edite `src/data_processing/data_loader.py`
2. **Novos Modelos**: Adicione em `src/models/`
3. **Visualizações**: Crie componentes em `dashboard/components/`
4. **API Endpoints**: Adicione em `src/api/app.py`

### Executar Notebooks Jupyter

```bash
jupyter notebook
```

Navegue até `notebooks/` e abra os notebooks de análise.

---

## 📝 Datasets

### Dataset Principal: `DATASET FINAL WRDP.csv`

- **Linhas**: ~10,000 registros (ajuste conforme seu dataset)
- **Colunas**: ~50 features
- **Target**: Diagnóstico (variável categórica)

### Formato Esperado:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Idade | int | Idade do paciente |
| Gênero | int/str | Gênero do paciente |
| Temperatura (°C) | float | Temperatura ambiente |
| Umidade | float | Umidade relativa do ar |
| Velocidade do Vento (km/h) | float | Velocidade do vento |
| [Sintomas] | int | 0 ou 1 (ausente/presente) |
| Diagnóstico | str | Doença diagnosticada |

---

## 🐛 Troubleshooting

### Erro: Modelo não encontrado
```bash
# Execute o script de treinamento
cd scripts
python train_models.py
```

### Erro: Módulo não encontrado
```bash
# Reinstale as dependências
pip install -r requirements.txt
```

### Dashboard não carrega
```bash
# Verifique se todas as dependências estão instaladas
pip list

# Verifique a porta
# O dashboard usa a porta 8050 por padrão
```

---

## 📚 Documentação Adicional

- [Documentação da API](docs/API.md) *(a criar)*
- [Guia de Modelos](docs/MODELS.md) *(a criar)*
- [Tutorial de Uso](docs/TUTORIAL.md) *(a criar)*

---

## 👥 Equipe

- **VitaNimbus Team**

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 Agradecimentos

- Dataset: Weather Related Disease Prediction
- Tecnologias: Scikit-learn, Dash, Plotly, Flask

---

## 📞 Contato

Para dúvidas ou sugestões, abra uma issue no GitHub ou entre em contato com a equipe.

---

**Desenvolvido com ❤️ pela equipe VitaNimbus**
