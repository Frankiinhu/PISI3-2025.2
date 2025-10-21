"""
Dashboard Principal - Nimbusvita (Versão Completa com Callbacks)
Análise Exploratória de Dados e Predição de Doenças Relacionadas ao Clima
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Variáveis globais que serão carregadas sob demanda
df_global = None
eda_global = None
classifier = None
clusterer = None
loader = None
symptom_cols = []
climatic_vars = []

def load_data_and_models():
    """Carrega dados e modelos apenas uma vez"""
    global df_global, eda_global, classifier, clusterer, loader, symptom_cols, climatic_vars
    
    if df_global is not None:
        return  # Já carregado
    
    print("Carregando dados...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DATASET FINAL WRDP.csv')
    loader = DataLoader(data_path)
    df_global = loader.get_clean_data()
    eda_global = ExploratoryDataAnalysis(df_global)
    
    # Obter feature names
    feature_dict = loader.get_feature_names()
    symptom_cols = feature_dict['symptoms']
    climatic_vars = feature_dict['climatic']
    
    # Carregar modelos
    print("Carregando modelos...")
    classifier = DiagnosisClassifier()
    clusterer = DiseaseClusterer()
    
    try:
        classifier_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'classifier_model.pkl')
        classifier.load_model(classifier_path)
        print("✓ Classificador carregado")
    except Exception as e:
        print(f"⚠ Classificador não carregado: {e}")
    
    try:
        clusterer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'clustering_model.pkl')
        clusterer.load_model(clusterer_path)
        print("✓ Clusterizador carregado")
    except Exception as e:
        print(f"⚠ Clusterizador não carregado: {e}")

# Inicializar app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Nimbusvita - Análise de Doenças Climáticas"

# CSS customizado para melhorar a aparência
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            /* Animações suaves */
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px rgba(0,0,0,0.6) !important;
            }
            
            .card-hover:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3) !important;
            }
            
            /* Estilo para inputs */
            input[type="number"] {
                background-color: rgba(255,255,255,0.05) !important;
                border: 1px solid #2d3250 !important;
                color: #e8eaf6 !important;
                transition: all 0.3s ease;
            }
            
            input[type="number"]:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
                outline: none;
            }
            
            /* Estilo para checkboxes */
            input[type="checkbox"] {
                accent-color: #667eea;
            }
            
            /* Animação do botão */
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
            }
            
            button:active {
                transform: translateY(0);
            }
            
            /* Scrollbar customizada */
            ::-webkit-scrollbar {
                width: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #0a0e27;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
            }
            
            /* Dropdown styles */
            .Select-control {
                background-color: rgba(255,255,255,0.05) !important;
                border-color: #2d3250 !important;
            }
            
            .Select-menu-outer {
                background-color: #1e2139 !important;
                border: 1px solid #2d3250 !important;
            }
            
            .Select-option {
                background-color: #1e2139 !important;
                color: #e8eaf6 !important;
            }
            
            .Select-option:hover {
                background-color: #252a48 !important;
            }
            
            /* Tabs animation */
            ._dash-undo-redo {
                display: none;
            }
            
            .tab {
                transition: all 0.3s ease;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Cores e estilo - Tema moderno e profissional
COLORS = {
    'background': '#0a0e27',
    'background_light': '#1a1f3a',
    'primary': '#667eea',
    'primary_light': '#764ba2',
    'secondary': '#131829',
    'text': '#e8eaf6',
    'text_secondary': '#9fa8da',
    'accent': '#f093fb',
    'accent_secondary': '#4facfe',
    'card': '#1e2139',
    'card_hover': '#252a48',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'error': '#f87171',
    'border': '#2d3250'
}

# Layout principal
app.layout = html.Div(style={
    'backgroundColor': COLORS['background'], 
    'minHeight': '100vh', 
    'fontFamily': "'Inter', 'Segoe UI', 'Roboto', sans-serif",
    'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["background_light"]} 100%)'
}, children=[
    # Header com gradiente
    html.Div(className='header', style={
        'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%)',
        'padding': '30px 20px',
        'marginBottom': '30px',
        'boxShadow': '0 10px 30px rgba(0,0,0,0.5)',
        'borderBottom': f'3px solid {COLORS["accent"]}'
    }, children=[
        html.Div(style={'maxWidth': '1400px', 'margin': '0 auto'}, children=[
            html.H1('Nimbusvita', 
                    style={
                        'color': 'white', 
                        'textAlign': 'center', 
                        'margin': '0', 
                        'fontSize': '3em',
                        'fontWeight': '700',
                        'letterSpacing': '1px',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
                    }),
            html.P('Weather Related Disease Prediction',
                   style={
                       'color': 'rgba(255,255,255,0.95)', 
                       'textAlign': 'center', 
                       'margin': '10px 0 5px 0', 
                       'fontSize': '1.3em',
                       'fontWeight': '500'
                   }),
            html.P('Análise Exploratória de Dados e Predição de Doenças Relacionadas ao Clima',
                   style={
                       'color': 'rgba(255,255,255,0.85)', 
                       'textAlign': 'center', 
                       'margin': '5px 0 0 0', 
                       'fontSize': '0.95em',
                       'fontWeight': '400'
                   })
        ])
    ]),
    
    # Container para tabs
    html.Div(style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '0 20px'}, children=[
        # Tabs de navegação com estilo moderno
        dcc.Tabs(id='tabs', value='tab-overview', style={
            'backgroundColor': 'transparent',
            'borderBottom': f'2px solid {COLORS["border"]}'
        }, children=[
            dcc.Tab(label='Visão Geral', value='tab-overview', 
                    style={
                        'color': COLORS['text_secondary'], 
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 30px',
                        'fontSize': '1em',
                        'fontWeight': '500',
                        'transition': 'all 0.3s ease'
                    },
                    selected_style={
                        'color': COLORS['accent'], 
                        'backgroundColor': COLORS['card'], 
                        'fontWeight': '600',
                        'borderTop': f'3px solid {COLORS["accent"]}',
                        'borderLeft': 'none',
                        'borderRight': 'none',
                        'borderBottom': 'none',
                        'borderRadius': '8px 8px 0 0'
                    }),
            dcc.Tab(label='Análise Exploratória', value='tab-eda', 
                    style={
                        'color': COLORS['text_secondary'], 
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 30px',
                        'fontSize': '1em',
                        'fontWeight': '500'
                    },
                    selected_style={
                        'color': COLORS['accent'], 
                        'backgroundColor': COLORS['card'], 
                        'fontWeight': '600',
                        'borderTop': f'3px solid {COLORS["accent"]}',
                        'borderLeft': 'none',
                        'borderRight': 'none',
                        'borderBottom': 'none',
                        'borderRadius': '8px 8px 0 0'
                    }),
            dcc.Tab(label='Clima vs Diagnóstico', value='tab-climate', 
                    style={
                        'color': COLORS['text_secondary'], 
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 30px',
                        'fontSize': '1em',
                        'fontWeight': '500'
                    },
                    selected_style={
                        'color': COLORS['accent'], 
                        'backgroundColor': COLORS['card'], 
                        'fontWeight': '600',
                        'borderTop': f'3px solid {COLORS["accent"]}',
                        'borderLeft': 'none',
                        'borderRight': 'none',
                        'borderBottom': 'none',
                        'borderRadius': '8px 8px 0 0'
                    }),
            dcc.Tab(label='Sintomas', value='tab-symptoms', 
                    style={
                        'color': COLORS['text_secondary'], 
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 30px',
                        'fontSize': '1em',
                        'fontWeight': '500'
                    },
                    selected_style={
                        'color': COLORS['accent'], 
                        'backgroundColor': COLORS['card'], 
                        'fontWeight': '600',
                        'borderTop': f'3px solid {COLORS["accent"]}',
                        'borderLeft': 'none',
                        'borderRight': 'none',
                        'borderBottom': 'none',
                        'borderRadius': '8px 8px 0 0'
                    }),
            dcc.Tab(label='Modelos ML', value='tab-ml', 
                    style={
                        'color': COLORS['text_secondary'], 
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 30px',
                        'fontSize': '1em',
                        'fontWeight': '500'
                    },
                    selected_style={
                        'color': COLORS['accent'], 
                        'backgroundColor': COLORS['card'], 
                        'fontWeight': '600',
                        'borderTop': f'3px solid {COLORS["accent"]}',
                        'borderLeft': 'none',
                        'borderRight': 'none',
                        'borderBottom': 'none',
                        'borderRadius': '8px 8px 0 0'
                    }),
            dcc.Tab(label='Predição', value='tab-prediction', 
                    style={
                        'color': COLORS['text_secondary'], 
                        'backgroundColor': 'transparent',
                        'border': 'none',
                        'padding': '15px 30px',
                        'fontSize': '1em',
                        'fontWeight': '500'
                    },
                    selected_style={
                        'color': COLORS['accent'], 
                        'backgroundColor': COLORS['card'], 
                        'fontWeight': '600',
                        'borderTop': f'3px solid {COLORS["accent"]}',
                        'borderLeft': 'none',
                        'borderRight': 'none',
                        'borderBottom': 'none',
                        'borderRadius': '8px 8px 0 0'
                    }),
        ]),
        
        # Conteúdo das tabs
        html.Div(id='tabs-content', style={'padding': '30px 0'})
    ])
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    """Renderiza conteúdo baseado na tab selecionada"""
    load_data_and_models()  # Garante que dados estão carregados
    
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


def create_card(children, title=None):
    """Cria um card estilizado com design moderno"""
    content = []
    if title:
        content.append(html.H3(title, style={
            'color': COLORS['text'], 
            'marginBottom': '20px',
            'fontSize': '1.3em',
            'fontWeight': '600',
            'borderBottom': f'2px solid {COLORS["accent"]}',
            'paddingBottom': '10px',
            'background': f'linear-gradient(90deg, {COLORS["accent"]} 0%, transparent 100%)',
            'WebkitBackgroundClip': 'text',
            'WebkitTextFillColor': 'transparent',
            'backgroundClip': 'text'
        }))
    content.extend(children if isinstance(children, list) else [children])
    
    return html.Div(
        content,
        style={
            'backgroundColor': COLORS['card'],
            'padding': '25px',
            'borderRadius': '15px',
            'marginBottom': '25px',
            'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLORS["border"]}',
            'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
            'position': 'relative',
            'overflow': 'hidden'
        },
        className='card-hover'
    )


def create_overview_layout():
    """Layout da visão geral"""
    info = eda_global.basic_info()
    
    # Estatísticas básicas com design moderno
    stats_cards = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div('📊', style={'fontSize': '2.5em', 'marginBottom': '10px'}),
                    html.H4('Total de Registros', style={
                        'color': COLORS['text_secondary'], 
                        'margin': '0', 
                        'fontSize': '0.9em',
                        'fontWeight': '500',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px'
                    }),
                    html.H2(f"{info['shape'][0]:,}", style={
                        'color': COLORS['accent'], 
                        'margin': '15px 0 0 0',
                        'fontSize': '2.5em',
                        'fontWeight': '700',
                        'background': f'linear-gradient(135deg, {COLORS["accent"]} 0%, {COLORS["accent_secondary"]} 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent'
                    }),
                ], style={
                    'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                    'padding': '30px 20px', 
                    'borderRadius': '15px', 
                    'textAlign': 'center', 
                    'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
                    'border': f'1px solid {COLORS["border"]}',
                    'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
                    'cursor': 'pointer'
                })
            ], style={'width': '100%'}, className='stat-card'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div('🏥', style={'fontSize': '2.5em', 'marginBottom': '10px'}),
                    html.H4('Diagnósticos Únicos', style={
                        'color': COLORS['text_secondary'], 
                        'margin': '0', 
                        'fontSize': '0.9em',
                        'fontWeight': '500',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px'
                    }),
                    html.H2(f"{df_global['Diagnóstico'].nunique()}", style={
                        'color': COLORS['success'], 
                        'margin': '15px 0 0 0',
                        'fontSize': '2.5em',
                        'fontWeight': '700'
                    }),
                ], style={
                    'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                    'padding': '30px 20px', 
                    'borderRadius': '15px',
                    'textAlign': 'center', 
                    'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
                    'border': f'1px solid {COLORS["border"]}',
                    'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
                    'cursor': 'pointer'
                })
            ], style={'width': '100%'}, className='stat-card'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div('📈', style={'fontSize': '2.5em', 'marginBottom': '10px'}),
                    html.H4('Total de Features', style={
                        'color': COLORS['text_secondary'], 
                        'margin': '0', 
                        'fontSize': '0.9em',
                        'fontWeight': '500',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px'
                    }),
                    html.H2(f"{info['shape'][1]}", style={
                        'color': COLORS['primary'], 
                        'margin': '15px 0 0 0',
                        'fontSize': '2.5em',
                        'fontWeight': '700'
                    }),
                ], style={
                    'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                    'padding': '30px 20px', 
                    'borderRadius': '15px',
                    'textAlign': 'center', 
                    'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
                    'border': f'1px solid {COLORS["border"]}',
                    'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
                    'cursor': 'pointer'
                })
            ], style={'width': '100%'}, className='stat-card'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div('🔬', style={'fontSize': '2.5em', 'marginBottom': '10px'}),
                    html.H4('Sintomas Analisados', style={
                        'color': COLORS['text_secondary'], 
                        'margin': '0', 
                        'fontSize': '0.9em',
                        'fontWeight': '500',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px'
                    }),
                    html.H2(f"{len(symptom_cols)}", style={
                        'color': COLORS['warning'], 
                        'margin': '15px 0 0 0',
                        'fontSize': '2.5em',
                        'fontWeight': '700'
                    }),
                ], style={
                    'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                    'padding': '30px 20px', 
                    'borderRadius': '15px',
                    'textAlign': 'center', 
                    'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
                    'border': f'1px solid {COLORS["border"]}',
                    'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
                    'cursor': 'pointer'
                })
            ], style={'width': '100%'}, className='stat-card'),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
    ], style={'marginBottom': '30px'})
    
    return html.Div([
        html.Div([
            html.H2('Visão Geral do Dataset', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Estatísticas e distribuições principais do conjunto de dados', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        stats_cards,
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='diagnosis-count-graph')], 'Distribuição de Diagnósticos')
            ], style={'width': '100%', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='age-distribution-graph')], 'Distribuição de Idade')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                create_card([dcc.Graph(id='climate-vars-distribution')], 'Variáveis Climáticas')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
    ])


@app.callback(
    Output('diagnosis-count-graph', 'figure'),
    Input('tabs', 'value')
)
def update_diagnosis_count(tab):
    """Atualiza gráfico de contagem de diagnósticos"""
    load_data_and_models()
    if tab != 'tab-overview':
        return {}
    
    diag_counts = df_global['Diagnóstico'].value_counts().reset_index()
    diag_counts.columns = ['Diagnóstico', 'Contagem']
    
    fig = px.bar(diag_counts, x='Diagnóstico', y='Contagem',
                 title='',
                 color='Contagem',
                 color_continuous_scale='Blues')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        xaxis_title='Diagnóstico',
        yaxis_title='Número de Casos',
        showlegend=False,
        xaxis_tickangle=-45,
        font=dict(family="Inter, sans-serif", size=12),
        title_font=dict(size=16, color=COLORS['text']),
        xaxis=dict(gridcolor=COLORS['border'], showgrid=True),
        yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
        margin=dict(t=30, b=80, l=60, r=30)
    )
    
    return fig


@app.callback(
    Output('age-distribution-graph', 'figure'),
    Input('tabs', 'value')
)
def update_age_distribution(tab):
    """Atualiza distribuição de idade"""
    load_data_and_models()
    if tab != 'tab-overview':
        return {}
    
    fig = px.histogram(df_global, x='Idade', nbins=30,
                      title='',
                      color_discrete_sequence=[COLORS['accent']])
    
    mean_age = df_global['Idade'].mean()
    fig.add_vline(x=mean_age, line_dash="dash", line_color="red",
                  annotation_text=f"Média: {mean_age:.1f} anos")
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Idade (anos)',
        yaxis_title='Frequência',
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('climate-vars-distribution', 'figure'),
    Input('tabs', 'value')
)
def update_climate_distribution(tab):
    """Atualiza distribuição de variáveis climáticas"""
    load_data_and_models()
    if tab != 'tab-overview':
        return {}
    
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=climatic_vars)
    
    colors = [COLORS['primary'], COLORS['accent'], '#FF6B6B']
    
    for i, var in enumerate(climatic_vars, 1):
        fig.add_trace(
            go.Histogram(x=df_global[var], name=var, marker_color=colors[i-1]),
            row=i, col=1
        )
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False
    )
    
    return fig


def create_eda_layout():
    """Layout da análise exploratória"""
    return html.Div([
        html.Div([
            html.H2('Análise Exploratória de Dados', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Explore correlações e padrões nos dados através de visualizações interativas', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        html.Div([
            html.Label('Selecione Sintomas para Análise:', 
                      style={
                          'color': COLORS['text'], 
                          'fontSize': '1em', 
                          'marginBottom': '12px',
                          'display': 'block',
                          'fontWeight': '600'
                      }),
            dcc.Dropdown(
                id='symptom-selector',
                options=[{'label': s, 'value': s} for s in symptom_cols[:20]],
                value=symptom_cols[:4] if len(symptom_cols) >= 4 else symptom_cols,
                multi=True,
                placeholder='Selecione sintomas...',
                style={
                    'backgroundColor': COLORS['secondary'], 
                    'color': COLORS['text'],
                    'borderRadius': '8px'
                }
            )
        ], style={
            'marginBottom': '30px', 
            'padding': '25px', 
            'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLORS["border"]}'
        }),
        
        html.Div([
            create_card([dcc.Graph(id='symptom-frequency-graphs')], 
                       'Frequência de Sintomas por Diagnóstico')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='correlation-matrix-graph')], 
                       'Matriz de Correlação (Top Features)')
        ]),
    ])


@app.callback(
    Output('symptom-frequency-graphs', 'figure'),
    [Input('symptom-selector', 'value'),
     Input('tabs', 'value')]
)
def update_symptom_frequency(selected_symptoms, tab):
    """Atualiza gráficos de frequência de sintomas"""
    load_data_and_models()
    if tab != 'tab-eda' or not selected_symptoms:
        return {}
    
    # Criar subplots
    n_symptoms = len(selected_symptoms)
    rows = (n_symptoms + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=selected_symptoms,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, symptom in enumerate(selected_symptoms):
        if symptom in df_global.columns:
            freq = df_global.groupby('Diagnóstico')[symptom].sum().reset_index()
            freq.columns = ['Diagnóstico', 'Contagem']
            
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fig.add_trace(
                go.Bar(x=freq['Diagnóstico'], y=freq['Contagem'], 
                      name=symptom, showlegend=False,
                      marker_color=COLORS['accent']),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * rows,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text']
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


@app.callback(
    Output('correlation-matrix-graph', 'figure'),
    Input('tabs', 'value')
)
def update_correlation_matrix(tab):
    """Atualiza matriz de correlação"""
    load_data_and_models()
    if tab != 'tab-eda':
        return {}
    
    # Selecionar top features
    if classifier.feature_importances is not None:
        top_features = classifier.feature_importances.head(15).index.tolist()
        # Adicionar variáveis climáticas
        features_to_correlate = list(set(top_features + climatic_vars + ['Idade']))
        features_to_correlate = [f for f in features_to_correlate if f in df_global.columns]
    else:
        features_to_correlate = climatic_vars + ['Idade'] + symptom_cols[:10]
    
    corr_matrix = df_global[features_to_correlate].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title='Correlação')
    ))
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_climate_layout():
    """Layout de análise climática"""
    return html.Div([
        html.Div([
            html.H2('Relação entre Clima e Diagnósticos', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Análise da influência das variáveis climáticas nos diferentes diagnósticos', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='temp-diagnosis-graph')], 
                       'Temperatura vs Diagnóstico')
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='humidity-diagnosis-graph')], 
                           'Umidade vs Diagnóstico')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                create_card([dcc.Graph(id='wind-diagnosis-graph')], 
                           'Velocidade do Vento vs Diagnóstico')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
    ])


@app.callback(
    Output('temp-diagnosis-graph', 'figure'),
    Input('tabs', 'value')
)
def update_temp_diagnosis(tab):
    """Atualiza gráfico temperatura vs diagnóstico"""
    load_data_and_models()
    if tab != 'tab-climate':
        return {}
    
    fig = px.box(df_global, x='Diagnóstico', y='Temperatura (°C)',
                 color='Diagnóstico',
                 title='')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig


@app.callback(
    Output('humidity-diagnosis-graph', 'figure'),
    Input('tabs', 'value')
)
def update_humidity_diagnosis(tab):
    """Atualiza gráfico umidade vs diagnóstico"""
    load_data_and_models()
    if tab != 'tab-climate':
        return {}
    
    fig = px.box(df_global, x='Diagnóstico', y='Umidade',
                 color='Diagnóstico',
                 title='')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig


@app.callback(
    Output('wind-diagnosis-graph', 'figure'),
    Input('tabs', 'value')
)
def update_wind_diagnosis(tab):
    """Atualiza gráfico vento vs diagnóstico"""
    load_data_and_models()
    if tab != 'tab-climate':
        return {}
    
    fig = px.box(df_global, x='Diagnóstico', y='Velocidade do Vento (km/h)',
                 color='Diagnóstico',
                 title='')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig


def create_symptoms_layout():
    """Layout de análise de sintomas"""
    return html.Div([
        html.Div([
            html.H2('Análise de Sintomas', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Mapeamento detalhado de sintomas e sua relação com diagnósticos', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='symptom-heatmap-graph')], 
                       'Heatmap de Sintomas por Diagnóstico (Top 15)')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='diagnosis-by-symptom-graph')], 
                       'Diagnóstico por Sintoma (Top 10 Sintomas)')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='symptom-importance-graph')], 
                       'Top 15 Sintomas por Importância')
        ]),
    ])


@app.callback(
    Output('symptom-heatmap-graph', 'figure'),
    Input('tabs', 'value')
)
def update_symptom_heatmap(tab):
    """Atualiza heatmap de sintomas"""
    load_data_and_models()
    if tab != 'tab-symptoms':
        return {}
    
    # Agrupar sintomas por diagnóstico
    symptom_by_diagnosis = df_global.groupby('Diagnóstico')[symptom_cols].sum()
    
    # Selecionar top 15 sintomas mais frequentes
    top_symptoms = symptom_by_diagnosis.sum().sort_values(ascending=False).head(15).index
    symptom_subset = symptom_by_diagnosis[top_symptoms]
    
    fig = go.Figure(data=go.Heatmap(
        z=symptom_subset.T.values,
        x=symptom_subset.index,
        y=top_symptoms,
        colorscale='Viridis',
        colorbar=dict(title='Contagem')
    ))
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_tickangle=-45
    )
    
    return fig


@app.callback(
    Output('diagnosis-by-symptom-graph', 'figure'),
    Input('tabs', 'value')
)
def update_diagnosis_by_symptom(tab):
    """Atualiza gráfico de diagnóstico por sintoma (inverso)"""
    load_data_and_models()
    if tab != 'tab-symptoms':
        return {}
    
    # Selecionar top 10 sintomas mais comuns
    symptom_totals = df_global[symptom_cols].sum().sort_values(ascending=False).head(10)
    top_10_symptoms = symptom_totals.index.tolist()
    
    # Criar subplots para cada sintoma
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=top_10_symptoms,
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, symptom in enumerate(top_10_symptoms):
        # Contar diagnósticos para pacientes com este sintoma
        patients_with_symptom = df_global[df_global[symptom] == 1]
        diagnosis_counts = patients_with_symptom['Diagnóstico'].value_counts()
        
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Bar(
                x=diagnosis_counts.index, 
                y=diagnosis_counts.values,
                name=symptom,
                showlegend=False,
                marker_color=colors[idx % len(colors)],
                text=diagnosis_counts.values,
                textposition='outside'
            ),
            row=row, col=col
        )
        
        # Ajustar ângulo dos labels do eixo x
        fig.update_xaxes(tickangle=-45, row=row, col=col)
    
    fig.update_layout(
        height=1200,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        title_text='Distribuição de Diagnósticos por Sintoma',
        title_x=0.5,
        title_font_size=16
    )
    
    return fig


@app.callback(
    Output('symptom-importance-graph', 'figure'),
    Input('tabs', 'value')
)
def update_symptom_importance(tab):
    """Atualiza gráfico de importância de sintomas"""
    load_data_and_models()
    if tab != 'tab-symptoms':
        return {}
    
    if classifier.feature_importances is None:
        return {}
    
    top_features = classifier.feature_importances.head(15)
    
    fig = px.bar(x=top_features.values, y=top_features.index,
                 orientation='h',
                 title='',
                 color=top_features.values,
                 color_continuous_scale='Blues')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Importância',
        yaxis_title='Feature',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_ml_layout():
    """Layout dos modelos de ML"""
    if classifier.model is None:
        return html.Div([
            html.Div([
                html.H2('Modelos de Machine Learning', style={
                    'color': COLORS['text'],
                    'fontSize': '2em',
                    'fontWeight': '700',
                    'marginBottom': '20px'
                }),
                html.Div([
                    html.P('⚠️ Modelos não carregados', style={
                        'color': COLORS['warning'],
                        'fontSize': '1.3em',
                        'fontWeight': '600',
                        'marginBottom': '10px'
                    }),
                    html.P('Execute o script de treinamento primeiro para visualizar os modelos.',
                          style={'color': COLORS['text_secondary'], 'fontSize': '1em'})
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': '30px',
                    'borderRadius': '15px',
                    'border': f'2px solid {COLORS["warning"]}',
                    'boxShadow': '0 8px 32px rgba(0,0,0,0.4)'
                })
            ])
        ])
    
    return html.Div([
        html.Div([
            html.H2('Modelos de Machine Learning', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Performance e visualizações dos modelos de classificação e clusterização', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='feature-importance-graph')], 
                       'Top 20 Features Mais Importantes (Random Forest)')
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='cluster-visualization-2d-graph')], 
                           'Visualização de Clusters (PCA 2D)')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                create_card([dcc.Graph(id='cluster-pca-3d-graph')], 
                           'Visualização de Clusters PCA 3D')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='cluster-visualization-3d-graph')], 
                           'Visualização de Clusters 3D (Variáveis Climáticas)')
            ], style={'width': '100%', 'padding': '10px'}),
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='classification-performance-graph')], 
                       'Performance da Classificação por Classe')
        ]),
    ])


@app.callback(
    Output('feature-importance-graph', 'figure'),
    Input('tabs', 'value')
)
def update_feature_importance(tab):
    """Atualiza gráfico de importância de features"""
    load_data_and_models()
    if tab != 'tab-ml' or classifier.feature_importances is None:
        return {}
    
    top_20 = classifier.feature_importances.head(20)
    
    fig = px.bar(x=top_20.values, y=top_20.index,
                 orientation='h',
                 title='',
                 color=top_20.values,
                 color_continuous_scale='Bluered')
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Importância',
        yaxis_title='Feature',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


@app.callback(
    Output('cluster-visualization-2d-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_visualization_2d(tab):
    """Atualiza visualização de clusters 2D"""
    load_data_and_models()
    if tab != 'tab-ml' or clusterer.labels_ is None:
        return {}
    
    # Preparar dados para visualização
    X_scaled = clusterer.scaler.transform(
        df_global.select_dtypes(include=[np.number]).drop('Diagnóstico', axis=1, errors='ignore')
    )
    
    # PCA para 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Criar DataFrame para plotar
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusterer.labels_.astype(str),
        'Diagnóstico': df_global['Diagnóstico'].values
    })
    
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                    hover_data=['Diagnóstico'],
                    title='',
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)',
        height=600
    )
    
    return fig


@app.callback(
    Output('cluster-pca-3d-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_pca_3d(tab):
    """Atualiza visualização de clusters em 3D com PCA"""
    load_data_and_models()
    if tab != 'tab-ml' or clusterer.labels_ is None:
        return {}
    
    # Preparar dados para visualização
    X_scaled = clusterer.scaler.transform(
        df_global.select_dtypes(include=[np.number]).drop('Diagnóstico', axis=1, errors='ignore')
    )
    
    # PCA para 3D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Criar DataFrame para plotar
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': clusterer.labels_.astype(str),
        'Diagnóstico': df_global['Diagnóstico'].values
    })
    
    fig = px.scatter_3d(
        plot_df, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Cluster',
        hover_data=['Diagnóstico'],
        title='',
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
        }
    )
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        height=600,
        scene=dict(
            xaxis=dict(
                title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)', 
                backgroundcolor=COLORS['background']
            ),
            yaxis=dict(
                title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)', 
                backgroundcolor=COLORS['background']
            ),
            zaxis=dict(
                title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var.)', 
                backgroundcolor=COLORS['background']
            ),
            bgcolor=COLORS['background']
        )
    )
    
    return fig


@app.callback(
    Output('cluster-visualization-3d-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_visualization_3d(tab):
    """Atualiza visualização de clusters em 3D com variáveis climáticas"""
    load_data_and_models()
    if tab != 'tab-ml' or clusterer.labels_ is None:
        return {}
    
    # Criar DataFrame para plotar com as 3 variáveis climáticas
    plot_df = pd.DataFrame({
        'Temperatura (°C)': df_global['Temperatura (°C)'],
        'Umidade': df_global['Umidade'],
        'Velocidade do Vento (km/h)': df_global['Velocidade do Vento (km/h)'],
        'Cluster': clusterer.labels_.astype(str),
        'Diagnóstico': df_global['Diagnóstico'].values,
        'Idade': df_global['Idade']
    })
    
    fig = px.scatter_3d(
        plot_df, 
        x='Temperatura (°C)', 
        y='Umidade', 
        z='Velocidade do Vento (km/h)',
        color='Cluster',
        hover_data=['Diagnóstico', 'Idade'],
        title='',
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={
            'Temperatura (°C)': 'Temperatura (°C)',
            'Umidade': 'Umidade (%)',
            'Velocidade do Vento (km/h)': 'Vento (km/h)'
        }
    )
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        height=600,
        scene=dict(
            xaxis=dict(title='Temperatura (°C)', backgroundcolor=COLORS['background']),
            yaxis=dict(title='Umidade (%)', backgroundcolor=COLORS['background']),
            zaxis=dict(title='Vento (km/h)', backgroundcolor=COLORS['background']),
            bgcolor=COLORS['background']
        )
    )
    
    return fig


@app.callback(
    Output('classification-performance-graph', 'figure'),
    Input('tabs', 'value')
)
def update_classification_performance(tab):
    """Atualiza gráfico de performance da classificação"""
    load_data_and_models()
    if tab != 'tab-ml' or classifier.model is None:
        return {}
    
    # Se houver métricas de validação disponíveis, usar
    # Caso contrário, usar feature importance como proxy
    if hasattr(classifier, 'cv_scores') and classifier.cv_scores:
        # Mostrar scores de cross-validation
        metrics_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(len(classifier.cv_scores))],
            'Accuracy': classifier.cv_scores
        })
        
        fig = px.bar(metrics_df, x='Fold', y='Accuracy',
                    title='',
                    color='Accuracy',
                    color_continuous_scale='Blues')
        
        fig.add_hline(y=np.mean(classifier.cv_scores), 
                     line_dash="dash", 
                     line_color="red",
                     annotation_text=f"Média: {np.mean(classifier.cv_scores):.3f}")
    else:
        # Mostrar distribuição de predições por classe
        from collections import Counter
        
        # Fazer predição em todo o dataset
        X = df_global[classifier.feature_names]
        predictions = classifier.predict(X)
        
        # Contar predições vs real
        pred_counts = pd.DataFrame({
            'Diagnóstico': df_global['Diagnóstico'].value_counts().index,
            'Real': df_global['Diagnóstico'].value_counts().values,
            'Predito': [Counter(predictions)[diag] for diag in df_global['Diagnóstico'].value_counts().index]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pred_counts['Diagnóstico'], y=pred_counts['Real'], 
                            name='Real', marker_color=COLORS['primary']))
        fig.add_trace(go.Bar(x=pred_counts['Diagnóstico'], y=pred_counts['Predito'], 
                            name='Predito', marker_color=COLORS['accent']))
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Diagnóstico',
        yaxis_title='Contagem' if not hasattr(classifier, 'cv_scores') else 'Accuracy',
        xaxis_tickangle=-45,
        barmode='group',
        height=500
    )
    
    return fig


def create_prediction_layout():
    """Layout de predição"""
    return html.Div([
        html.Div([
            html.H2('Sistema de Predição de Diagnóstico', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Utilize inteligência artificial para prever diagnósticos baseados em dados clínicos e climáticos', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        create_card([
            html.Div([
                html.Div('🔮', style={
                    'fontSize': '3em',
                    'textAlign': 'center',
                    'marginBottom': '15px'
                }),
                html.P('Insira os dados do paciente para obter uma predição de diagnóstico:',
                      style={
                          'color': COLORS['text'], 
                          'fontSize': '1.1em', 
                          'marginBottom': '30px',
                          'textAlign': 'center',
                          'fontWeight': '500'
                      })
            ]),
            
            # Inputs climáticos e demográficos
            html.Div([
                html.Div([
                    html.Label('Idade:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-age', type='number', value=35,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Temperatura (°C):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-temp', type='number', value=25.0, step=0.1,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Umidade (0-1):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-humidity', type='number', value=0.7, step=0.01, min=0, max=1,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Velocidade do Vento (km/h):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-wind', type='number', value=10.0, step=0.1,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
            ]),
            
            # Seleção de sintomas
            html.Div([
                html.Label('Selecione Sintomas Presentes:', 
                          style={'color': COLORS['text'], 'fontSize': '1.1em', 
                                'fontWeight': 'bold', 'marginTop': '20px'}),
                dcc.Checklist(
                    id='symptom-checklist',
                    options=[{'label': f' {s}', 'value': s} for s in symptom_cols[:30]],
                    value=[],
                    style={'color': COLORS['text'], 'columnCount': 3, 'marginTop': '15px'},
                    labelStyle={'display': 'block', 'marginBottom': '10px'}
                )
            ], style={'marginTop': '20px'}),
            
            html.Div([
                html.Button('🔍 Fazer Predição', id='predict-button', 
                           style={
                               'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%)',
                               'color': 'white',
                               'padding': '18px 50px',
                               'border': 'none',
                               'borderRadius': '50px',
                               'fontSize': '1.2em',
                               'cursor': 'pointer',
                               'fontWeight': '700',
                               'boxShadow': '0 10px 30px rgba(102, 126, 234, 0.4)',
                               'transition': 'all 0.3s ease',
                               'textTransform': 'uppercase',
                               'letterSpacing': '1px',
                               'width': '100%',
                               'maxWidth': '400px'
                           })
            ], style={'textAlign': 'center', 'marginTop': '40px'}),
            
            html.Div(id='prediction-result', style={'marginTop': '30px'})
        ])
    ])


@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('input-age', 'value'),
     State('input-temp', 'value'),
     State('input-humidity', 'value'),
     State('input-wind', 'value'),
     State('symptom-checklist', 'value')]
)
def make_prediction(n_clicks, age, temp, humidity, wind, symptoms):
    """Faz predição de diagnóstico"""
    load_data_and_models()
    if n_clicks is None or classifier.model is None:
        return html.Div()
    
    try:
        # Construir features
        features = {
            'Idade': age or 35,
            'Temperatura (°C)': temp or 25.0,
            'Umidade': humidity or 0.7,
            'Velocidade do Vento (km/h)': wind or 10.0
        }
        
        # Adicionar sintomas
        for symptom in classifier.feature_names:
            if symptom not in features:
                features[symptom] = 1 if symptom in (symptoms or []) else 0
        
        # Criar DataFrame
        X = pd.DataFrame([features])
        X = X[classifier.feature_names]
        
        # Fazer predição
        diagnosis = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]
        confidence = max(probabilities) * 100
        
        # Preparar resultado
        all_probs = sorted(zip(classifier.label_encoder.classes_, probabilities), 
                          key=lambda x: x[1], reverse=True)
        
        result = html.Div([
            html.Div([
                html.Div([
                    html.Div('✨', style={'fontSize': '3em', 'textAlign': 'center', 'marginBottom': '15px'}),
                    html.H3('Resultado da Predição', style={
                        'color': COLORS['text'], 
                        'marginBottom': '30px',
                        'textAlign': 'center',
                        'fontSize': '1.8em',
                        'fontWeight': '700'
                    })
                ]),
                
                html.Div([
                    html.Div([
                        html.H4('Diagnóstico Predito', style={
                            'color': COLORS['text_secondary'], 
                            'margin': '0 0 15px 0',
                            'fontSize': '0.9em',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'fontWeight': '600'
                        }),
                        html.H2(diagnosis, style={
                            'color': COLORS['accent'], 
                            'margin': '10px 0 20px 0',
                            'fontSize': '2.5em',
                            'fontWeight': '700',
                            'background': f'linear-gradient(135deg, {COLORS["accent"]} 0%, {COLORS["accent_secondary"]} 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent'
                        }),
                        html.Div([
                            html.Span('Confiança: ', style={
                                'color': COLORS['text_secondary'],
                                'fontSize': '1em',
                                'fontWeight': '500'
                            }),
                            html.Span(f'{confidence:.2f}%', style={
                                'color': COLORS['success'] if confidence > 70 else COLORS['warning'],
                                'fontSize': '1.5em',
                                'fontWeight': '700'
                            })
                        ])
                    ], style={'textAlign': 'center'})
                ], style={
                    'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["background_light"]} 100%)',
                    'padding': '30px', 
                    'borderRadius': '15px', 
                    'marginBottom': '30px',
                    'border': f'2px solid {COLORS["accent"]}',
                    'boxShadow': f'0 0 30px rgba(240, 147, 251, 0.3)'
                }),
                
                html.Div([
                    html.H4('Probabilidades por Diagnóstico', style={
                        'color': COLORS['text'], 
                        'marginBottom': '20px',
                        'fontSize': '1.2em',
                        'fontWeight': '600',
                        'borderBottom': f'2px solid {COLORS["border"]}',
                        'paddingBottom': '10px'
                    }),
                    
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span(diag, style={
                                    'fontWeight': '600', 
                                    'color': COLORS['text'],
                                    'fontSize': '1em'
                                }),
                                html.Div([
                                    html.Div(style={
                                        'width': f'{prob*100}%',
                                        'height': '8px',
                                        'background': f'linear-gradient(90deg, {COLORS["accent"]} 0%, {COLORS["accent_secondary"]} 100%)' if prob == max(probabilities) else f'linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%)',
                                        'borderRadius': '4px',
                                        'transition': 'width 0.5s ease'
                                    })
                                ], style={
                                    'backgroundColor': COLORS['background'],
                                    'borderRadius': '4px',
                                    'marginTop': '8px',
                                    'marginBottom': '8px'
                                }),
                                html.Span(f'{prob*100:.2f}%', style={
                                    'color': COLORS['accent'] if prob == max(probabilities) else COLORS['text_secondary'],
                                    'fontSize': '0.9em',
                                    'fontWeight': '700'
                                })
                            ], style={
                                'padding': '15px',
                                'backgroundColor': COLORS['background_light'],
                                'borderRadius': '10px',
                                'marginBottom': '12px',
                                'border': f'2px solid {COLORS["accent"]}' if prob == max(probabilities) else f'1px solid {COLORS["border"]}',
                                'boxShadow': f'0 4px 15px rgba(240, 147, 251, 0.3)' if prob == max(probabilities) else '0 2px 8px rgba(0,0,0,0.2)'
                            })
                        ])
                        for diag, prob in all_probs[:5]
                    ])
                ])
            ], style={
                'backgroundColor': COLORS['card'], 
                'padding': '40px', 
                'borderRadius': '20px', 
                'boxShadow': '0 10px 40px rgba(0,0,0,0.5)',
                'border': f'1px solid {COLORS["border"]}'
            })
        ])
        
        return result
        
    except Exception as e:
        return html.Div([
            html.P(f'Erro na predição: {str(e)}', 
                  style={'color': 'red', 'fontSize': '1.1em'})
        ])


if __name__ == '__main__':
    print("\n" + "="*70)
    print("✨ NIMBUSVITA DASHBOARD")
    print("="*70)
    print("🚀 Dashboard iniciado com sucesso!")
    print("🌐 Acesse: http://127.0.0.1:8050/")
    print("📊 Sistema de Predição de Doenças Relacionadas ao Clima")
    print("="*70 + "\n")
    app.run(debug=True, port=8050)
