🌡️ NimbusVita - Weather Related Disease Prediction

NimbusVita combina análise exploratória, modelagem de aprendizado de máquina e visualizações interativas em Dash para investigar doenças relacionadas ao clima. O fluxo atual prioriza dois componentes principais:

- Treinamento de modelos de classificação e clusterização a partir do dataset `DATASET FINAL WRDP.csv`.
- Dashboard interativo (`dashboard/app_complete.py`) com métricas, PCA 3D, análise de clusters (k fixo e dinâmico) e visões climáticas.

� Estrutura Principal

```
PISI3-2025.2/
├── data/                     # Arquivos de dados (inclui DATASET FINAL WRDP.csv)
├── dashboard/                # Aplicação Dash
│   ├── app_complete.py       # Entry point do dashboard
│   ├── components.py         # Utilidades de layout
│   ├── core/                 # Contexto e tema
│   ├── models/saved_models/  # Modelos pré-treinados (.pkl)
│   └── views/                # Abas e callbacks
├── scripts/
│   └── train_models.py       # Pipeline CLI para treinar e salvar modelos
├── src/
│   ├── data_processing/      # DataLoader e EDA helpers
│   └── models/               # Implementações de classificação/clusterização
├── requirements.txt
└── README.md
```

�️ Preparação do Ambiente

1. Clonar o repositório
   ```bash
   git clone https://github.com/Frankiinhu/PISI3-2025.2.git
   cd PISI3-2025.2
   ```

2. Criar e ativar um ambiente virtual
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instalar dependências
   ```bash
   pip install -r requirements.txt
   ```

📈 Treinamento dos Modelos

Certifique-se de que o dataset esteja em `data/DATASET FINAL WRDP.csv`. Em seguida, execute:

```bash
python scripts/train_models.py --data data/DATASET FINAL WRDP.csv \
    --out-dir dashboard/models/saved_models \
    --classifier-name classifier_model.pkl \
    --clusterer-name clustering_model.pkl
```

O comando salva os artefatos esperados pelo dashboard em `dashboard/models/saved_models/`. Utilize as flags `--skip-classifier` ou `--skip-clusterer` se quiser treinar um modelo por vez.

� Executando o Dashboard

Com os modelos treinados, rode:

```bash
python -m dashboard.app_complete
```

O Dash sobe em `http://127.0.0.1:8050`. As principais seções hoje incluem:

- **Visão Geral** com métricas treinadas e cartões de status.
- **Análise Exploratório** com filtros demográficos/climáticos e gráficos dinâmicos.
- **Modelos de ML** com PCA 3D (k definido via elbow), comparação de métricas, clusters climáticos com slider (k=3–7) e barras empilhadas (k fixo 6 vs k elbow).

📚 Scripts Auxiliares

- `scripts/advanced_analysis.py`: gera visualizações estáticas (matplotlib/seaborn) adicionais e relatórios CSV.

📂 Formato do Dataset

O `DataLoader` espera colunas com nomes no padrão do arquivo oficial, incluindo:

- Variáveis climáticas: `Temperatura (°C)`, `Umidade`, `Velocidade do Vento (km/h)`.
- Dados demográficos: `Idade`, `Gênero`.
- Sintomas binários (0/1).
- Coluna alvo `Diagnóstico`.

� Solução de Problemas

- **Modelos não carregam no dashboard**: verifique se os arquivos `.pkl` atualizados estão em `dashboard/models/saved_models/` e reinicie a aplicação.
- **Erros de dependência**: confirme que o ambiente virtual está ativo e reinstale com `pip install -r requirements.txt`.
- **Dataset diferente**: ajuste os nomes das colunas ou atualize `DataLoader` para refletir o novo formato.

📄 Licença

Projeto distribuído sob a licença MIT. Contributions e sugestões são bem-vindas! Desenvolvido com ❤️ pela equipe NimbusVita.