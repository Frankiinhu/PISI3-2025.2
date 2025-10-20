"""
Dashboard Principal - VitaNimbus
Análise Exploratória de Dados e Predição de Doenças Relacionadas ao Clima
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# Adicionar diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import DataLoader
from src.data_processing.eda import ExploratoryDataAnalysis
from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer

# Inicializar app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "VitaNimbus - Análise de Doenças Climáticas"

# Cores e estilo
COLORS = {
    'background': '#0f1419',
    'primary': '#1DA1F2',
    'secondary': '#14171A',
    'text': '#E1E8ED',
    'accent': '#00C9A7'
}

# Layout principal
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'}, children=[
    # Header
    html.Div(className='header', style={
        'backgroundColor': COLORS['secondary'],
        'padding': '20px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }, children=[
        html.H1('🌡️ VitaNimbus - Weather Related Disease Prediction', 
                style={'color': COLORS['primary'], 'textAlign': 'center', 'margin': '0'}),
        html.P('Análise Exploratória de Dados e Predição de Doenças Relacionadas ao Clima',
               style={'color': COLORS['text'], 'textAlign': 'center', 'margin': '10px 0 0 0'})
    ]),
    
    # Tabs de navegação
    dcc.Tabs(id='tabs', value='tab-overview', style={
        'backgroundColor': COLORS['secondary']
    }, children=[
        dcc.Tab(label='📊 Visão Geral', value='tab-overview', style={'color': COLORS['text']},
                selected_style={'color': COLORS['primary']}),
        dcc.Tab(label='🔍 Análise Exploratória', value='tab-eda', style={'color': COLORS['text']},
                selected_style={'color': COLORS['primary']}),
        dcc.Tab(label='🌡️ Clima vs Diagnóstico', value='tab-climate', style={'color': COLORS['text']},
                selected_style={'color': COLORS['primary']}),
        dcc.Tab(label='💊 Sintomas', value='tab-symptoms', style={'color': COLORS['text']},
                selected_style={'color': COLORS['primary']}),
        dcc.Tab(label='🤖 Modelos ML', value='tab-ml', style={'color': COLORS['text']},
                selected_style={'color': COLORS['primary']}),
        dcc.Tab(label='🎯 Predição', value='tab-prediction', style={'color': COLORS['text']},
                selected_style={'color': COLORS['primary']}),
    ]),
    
    # Conteúdo das tabs
    html.Div(id='tabs-content', style={'padding': '20px'})
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    """Renderiza conteúdo baseado na tab selecionada"""
    
    if tab == 'tab-overview':
        return create_overview_layout()
    elif tab == 'tab-eda':
        return create_eda_layout()
    elif tab == 'tab-climate':
        return create_climate_layout()
    elif tab == 'tab-symptoms':
        return create_symptoms_layout()
    elif tab == 'tab-ml':
        return create_ml_layout()
    elif tab == 'tab-prediction':
        return create_prediction_layout()


def create_overview_layout():
    """Layout da visão geral"""
    return html.Div([
        html.H2('📊 Visão Geral do Dataset', style={'color': COLORS['primary']}),
        
        html.Div(id='overview-stats', style={'marginTop': '20px'}),
        
        html.Div([
            html.Div([
                html.H3('Contagem de Diagnósticos', style={'color': COLORS['text']}),
                dcc.Graph(id='diagnosis-count-graph')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3('Distribuição de Idade', style={'color': COLORS['text']}),
                dcc.Graph(id='age-distribution-graph')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.H3('Distribuição de Variáveis Climáticas', style={'color': COLORS['text']}),
            dcc.Graph(id='climate-distribution-graph')
        ], style={'marginTop': '20px'})
    ])


def create_eda_layout():
    """Layout da análise exploratória"""
    return html.Div([
        html.H2('🔍 Análise Exploratória de Dados', style={'color': COLORS['primary']}),
        
        html.Div([
            html.Label('Selecione Sintomas para Análise:', style={'color': COLORS['text']}),
            dcc.Dropdown(
                id='symptom-selector',
                multi=True,
                placeholder='Selecione sintomas...',
                style={'backgroundColor': COLORS['secondary'], 'color': COLORS['text']}
            )
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H3('Frequência de Sintomas por Diagnóstico', style={'color': COLORS['text']}),
            html.Div(id='symptom-frequency-graphs')
        ]),
        
        html.Div([
            html.H3('Matriz de Correlação', style={'color': COLORS['text']}),
            dcc.Graph(id='correlation-matrix-graph')
        ], style={'marginTop': '20px'}),
    ])


def create_climate_layout():
    """Layout de análise climática"""
    return html.Div([
        html.H2('🌡️ Relação entre Clima e Diagnósticos', style={'color': COLORS['primary']}),
        
        html.Div([
            html.Div([
                html.H3('Temperatura vs Diagnóstico', style={'color': COLORS['text']}),
                dcc.Graph(id='temp-diagnosis-graph')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3('Umidade vs Diagnóstico', style={'color': COLORS['text']}),
                dcc.Graph(id='humidity-diagnosis-graph')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.H3('Velocidade do Vento vs Diagnóstico', style={'color': COLORS['text']}),
            dcc.Graph(id='wind-diagnosis-graph')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H3('Análise Multivariada Climática', style={'color': COLORS['text']}),
            dcc.Graph(id='climate-multivariate-graph')
        ], style={'marginTop': '20px'})
    ])


def create_symptoms_layout():
    """Layout de análise de sintomas"""
    return html.Div([
        html.H2('💊 Análise de Sintomas', style={'color': COLORS['primary']}),
        
        html.Div([
            html.H3('Heatmap de Sintomas por Diagnóstico', style={'color': COLORS['text']}),
            dcc.Graph(id='symptom-heatmap-graph')
        ]),
        
        html.Div([
            html.H3('Top Sintomas por Importância', style={'color': COLORS['text']}),
            dcc.Graph(id='symptom-importance-graph')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H3('Clima vs Sintomas Selecionados', style={'color': COLORS['text']}),
            html.Div(id='climate-symptom-graphs')
        ], style={'marginTop': '20px'})
    ])


def create_ml_layout():
    """Layout dos modelos de ML"""
    return html.Div([
        html.H2('🤖 Modelos de Machine Learning', style={'color': COLORS['primary']}),
        
        html.Div([
            html.H3('Classificação - Métricas do Modelo', style={'color': COLORS['text']}),
            html.Div(id='classification-metrics')
        ]),
        
        html.Div([
            html.H3('Importância das Features', style={'color': COLORS['text']}),
            dcc.Graph(id='feature-importance-graph')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H3('Matriz de Confusão', style={'color': COLORS['text']}),
            dcc.Graph(id='confusion-matrix-graph')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H3('Análise de Clusters', style={'color': COLORS['text']}),
            dcc.Graph(id='cluster-visualization-graph')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H3('Fatores de Risco por Cluster', style={'color': COLORS['text']}),
            html.Div(id='risk-factors-content')
        ], style={'marginTop': '20px'})
    ])


def create_prediction_layout():
    """Layout de predição"""
    return html.Div([
        html.H2('🎯 Sistema de Predição de Diagnóstico', style={'color': COLORS['primary']}),
        
        html.Div([
            html.P('Insira os dados do paciente para predição:', style={'color': COLORS['text']}),
            
            # Inputs para features
            html.Div([
                html.Div([
                    html.Label('Idade:', style={'color': COLORS['text']}),
                    dcc.Input(id='input-age', type='number', placeholder='Idade', 
                             style={'width': '100%', 'padding': '10px'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Temperatura (°C):', style={'color': COLORS['text']}),
                    dcc.Input(id='input-temp', type='number', placeholder='Temperatura', 
                             style={'width': '100%', 'padding': '10px'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Umidade (%):', style={'color': COLORS['text']}),
                    dcc.Input(id='input-humidity', type='number', placeholder='Umidade', 
                             style={'width': '100%', 'padding': '10px'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Velocidade do Vento (km/h):', style={'color': COLORS['text']}),
                    dcc.Input(id='input-wind', type='number', placeholder='Vento', 
                             style={'width': '100%', 'padding': '10px'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
            ]),
            
            html.Div([
                html.Label('Selecione Sintomas Presentes:', style={'color': COLORS['text']}),
                dcc.Checklist(
                    id='symptom-checklist',
                    options=[],
                    value=[],
                    style={'color': COLORS['text'], 'columnCount': 3}
                )
            ], style={'marginTop': '20px'}),
            
            html.Button('Fazer Predição', id='predict-button', 
                       style={
                           'backgroundColor': COLORS['primary'],
                           'color': 'white',
                           'padding': '15px 30px',
                           'border': 'none',
                           'borderRadius': '5px',
                           'fontSize': '16px',
                           'cursor': 'pointer',
                           'marginTop': '20px'
                       }),
            
            html.Div(id='prediction-result', style={'marginTop': '20px'})
        ])
    ])


if __name__ == '__main__':
    app.run(debug=True, port=8050)
