#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemplo de Uso da Aba "Visão Geral" (Overview)
Demonstra como o dashboard funciona com filtros e gráficos
"""

# ============================================================================
# INSTALAÇÃO DE DEPENDÊNCIAS
# ============================================================================
"""
Certifique-se de ter instalado:

pip install -r requirements.txt

Ou manualmente:
pip install dash==3.2.0
pip install plotly==6.3.1
pip install pandas==2.3.3
pip install numpy==2.3.4
pip install scikit-learn==1.7.2
pip install dash-bootstrap-components==1.7.0
"""

# ============================================================================
# EXEMPLO 1: Executar o Dashboard Completo
# ============================================================================
"""
1. Abra um terminal na pasta do projeto:
   cd c:\Users\Rubens\PISI3-2025.2

2. Execute o dashboard:
   python -m dashboard.app_complete

3. Acesse no navegador:
   http://127.0.0.1:8050/

4. Clique na aba "Visão Geral" para ver as novas funcionalidades
"""

# ============================================================================
# EXEMPLO 2: Componentes Disponíveis
# ============================================================================

from dashboard.views.overview import (
    create_layout,
    register_callbacks,
    _filter_dropdown,
    _kpi_card,
    _alert_component
)
from dashboard.core.theme import COLORS

# Exemplo: Criar um KPI card
kpi_example = _kpi_card(
    icon='📊',
    label='Total de Casos',
    value='5,200',
    value_color=COLORS['accent'],
    subtitle='Registros no dataset'
)

# Exemplo: Criar um alerta
alert_example = _alert_component(
    alert_type='success',
    title='Dados Balanceados',
    message='2,500 registros com distribuição adequada para análise.'
)

# ============================================================================
# EXEMPLO 3: Estrutura dos Filtros
# ============================================================================
"""
FILTRO DE GÊNERO:
  - 1: Masculino (👨)
  - 0: Feminino (👩)
  - 'todos': Sem filtro

FILTRO DE IDADE:
  - 'crianca': 0-12 anos (👶)
  - 'adolescente': 13-17 anos (🧒)
  - 'adulto': 18-59 anos (👨)
  - 'idoso': 60+ anos (👴)
  - 'todos': Sem filtro
"""

# ============================================================================
# EXEMPLO 4: Gráficos Disponíveis
# ============================================================================
"""
Os seguintes gráficos estão implementados na aba:

1. diagnosis-count-graph
   - Tipo: Gráfico de Barras (Plotly)
   - Filtros: Gênero, Idade
   - Dados: Contagem de diagnósticos

2. gender-pie-chart
   - Tipo: Gráfico de Pizza (Plotly)
   - Filtros: Idade
   - Dados: Distribuição de gênero

3. age-dist-univariate
   - Tipo: Histograma (Plotly)
   - Filtros: Gênero
   - Dados: Distribuição de idade com linhas de média/mediana

4. age-gender-heatmap
   - Tipo: Mapa de Calor (Plotly)
   - Filtros: Gênero, Idade
   - Dados: Cruzamento idade vs diagnóstico

5. climate-vars-distribution
   - Tipo: Multi-Histograma (Plotly)
   - Filtros: Gênero, Idade
   - Dados: Temperatura, Umidade, Vento

6. diagnosis-age-violin
   - Tipo: Gráfico Violino (Plotly)
   - Filtros: Gênero, Idade
   - Dados: Distribuição de diagnóstico por idade

7. top-diagnoses-by-gender
   - Tipo: Barras Agrupadas (Plotly)
   - Filtros: Idade
   - Dados: Top 8 diagnósticos por gênero

8. overview-alerts-container
   - Tipo: Sistema de Alertas
   - Filtros: Gênero, Idade
   - Gerado dinamicamente baseado em análise de dados
"""

# ============================================================================
# EXEMPLO 5: Callbacks (Atualizações em Tempo Real)
# ============================================================================
"""
Todos os gráficos atualizam quando você:

1. Seleciona uma opção de GÊNERO:
   - Todos os 7 gráficos se atualizam
   - Alertas são recalculados

2. Seleciona uma opção de IDADE:
   - Todos os 7 gráficos se atualizam
   - Alertas são recalculados

3. Muda de aba:
   - Callbacks só executam para tab='tab-overview'
   - Performance otimizada
"""

# ============================================================================
# EXEMPLO 6: Estrutura de Dados Esperada
# ============================================================================
"""
Dataset deve conter estas colunas:

├── Identificadores
│   └── ID do registro

├── Demográficos
│   ├── Gênero (0=Feminino, 1=Masculino)
│   ├── Idade (0-120 anos)
│   └── Localização

├── Clínicos
│   ├── Diagnóstico (H1, H2, H3, etc.)
│   ├── Sintomas (multi-coluna)
│   └── Data de diagnóstico

└── Climáticos
    ├── Temperatura (°C)
    ├── Umidade (%)
    └── Velocidade do Vento (km/h)

Exemplo de linha:
ID | Gênero | Idade | Diagnóstico | Temp | Umidade | Vento
1  |   0    |  25   |     H1      | 28.5 |   65    |  12.3
2  |   1    |  45   |     H2      | 29.1 |   60    |  11.8
"""

# ============================================================================
# EXEMPLO 7: Testes e Validação
# ============================================================================
"""
Executar testes:
python test_overview_tab.py

Verificar import:
python -c "from dashboard.views import overview; print('✓ OK')"

Validar componentes:
python -c "import dash_bootstrap_components; print('✓ Bootstrap OK')"
"""

# ============================================================================
# EXEMPLO 8: Personalizações Possíveis
# ============================================================================
"""
Você pode customizar a aba editando:

1. Cores (dashboard/core/theme.py):
   COLORS['primary'] = '#seu_hexcode'

2. Ícones (overview.py, create_layout):
   _kpi_card(icon='🆕', ...)

3. Títulos (overview.py, create_layout):
   html.H3('Seu Novo Título', ...)

4. Gráficos (overview.py, update_* functions):
   fig.update_layout(...)

5. Filtros (overview.py, create_layout):
   _filter_dropdown(..., options=[...])
"""

# ============================================================================
# EXEMPLO 9: Troubleshooting
# ============================================================================
"""
PROBLEMA: ModuleNotFoundError: No module named 'dash_bootstrap_components'
SOLUÇÃO: pip install dash-bootstrap-components

PROBLEMA: Dashboard não carrega dados
SOLUÇÃO: Verifique se o arquivo 'DATASET FINAL WRDP.csv' existe

PROBLEMA: Gráficos aparecem em branco
SOLUÇÃO: Aguarde carregamento, verifique console para erros

PROBLEMA: Filtros não atualizam gráficos
SOLUÇÃO: Certifique-se que está na aba 'tab-overview'

PROBLEMA: Performance lenta
SOLUÇÃO: Reduzir tamanho do dataset ou usar cache
"""

# ============================================================================
# EXEMPLO 10: Próximos Passos
# ============================================================================
"""
Para melhorar ainda mais a aba, considere:

1. Adicionar filtro por diagnóstico específico
2. Implementar export de dados em CSV
3. Criar comparações temporais
4. Adicionar análise de correlação
5. Implementar dashboard em tempo real com WebSocket
6. Adicionar drill-down nos gráficos
7. Criar relatórios automáticos
8. Integrar com banco de dados
"""

# ============================================================================
# EXECUTAR EXEMPLO
# ============================================================================

if __name__ == '__main__':
    print("""
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║      EXEMPLO DE USO - Aba "Visão Geral" do NimbusVita        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Para executar o dashboard:
    
    1. Terminal:
       cd c:\\Users\\Rubens\\PISI3-2025.2
       python -m dashboard.app_complete
    
    2. Navegador:
       http://127.0.0.1:8050/
    
    3. Clique em "Visão Geral" para ver as melhorias
    
    ═══════════════════════════════════════════════════════════════
    
    Componentes disponíveis:
    ✓ 4 KPIs responsivos
    ✓ 2 filtros interativos
    ✓ Alertas automáticos
    ✓ 7 gráficos avançados
    ✓ Layout responsivo (mobile, tablet, desktop)
    
    ═══════════════════════════════════════════════════════════════
    
    Teste a aba agora! 🚀
    """)
