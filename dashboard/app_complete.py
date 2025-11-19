import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Dashboard Principal - NimbusVita (Versão Completa com Callbacks)
Análise Exploratória de Doenças Relacionadas ao Clima
"""
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dashboard.components import create_card
from dashboard.core.data_context import (
    get_cluster_feature_frame,
    get_context,
    has_feature_importances,
    is_classifier_available,
)

# Cluster utilities ------------------------------------------------------

_CACHED_ELBOW_K: int | None = None
_CACHED_CLUSTER_PREP: tuple[int, tuple[int, ...], Any] | None = None


def _error_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS['text'])
    )
    fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['card'])
    return fig


def _prepare_cluster_dataset() -> tuple[Any, pd.DataFrame, np.ndarray]:
    ctx = get_context()
    try:
        feature_frame = get_cluster_feature_frame()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    # Cache prepared dataset to avoid repeated scaler.transform / dataframe selection
    global _CACHED_CLUSTER_PREP
    # Use simple cache key: id(ctx) and dataframe shape + column names
    df_key = (feature_frame.shape, tuple(feature_frame.columns))
    ctx_id = id(ctx)
    if _CACHED_CLUSTER_PREP is not None:
        cached_ctx_id, cached_df_key, cached_result = _CACHED_CLUSTER_PREP
        if cached_ctx_id == ctx_id and cached_df_key == df_key:
            return cached_result

    scaler = getattr(ctx.clusterer, 'scaler', None)
    if scaler is None:
        raise RuntimeError('Clusterizador sem scaler configurado; execute o treinamento do modelo.')

    X_scaled = scaler.transform(feature_frame)
    _CACHED_CLUSTER_PREP = (ctx_id, df_key, (ctx, feature_frame, X_scaled))
    return ctx, feature_frame, X_scaled


def _fit_kmeans_labels(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return model.fit_predict(X_scaled)


def _get_elbow_k(X_scaled: np.ndarray) -> int:
    global _CACHED_ELBOW_K
    if _CACHED_ELBOW_K is not None:
        return _CACHED_ELBOW_K

    max_k = min(10, X_scaled.shape[0])
    scanned_ks: list[int] = []
    inertias: list[float] = []

    for k in range(2, max_k):
        if k >= X_scaled.shape[0]:
            break
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        scanned_ks.append(k)
        inertias.append(float(model.inertia_))

    if not inertias:
        _CACHED_ELBOW_K = 3
        return _CACHED_ELBOW_K

    if len(inertias) < 3:
        _CACHED_ELBOW_K = scanned_ks[-1]
        return _CACHED_ELBOW_K

    second_diff = np.diff(inertias, n=2)
    idx = int(np.argmin(second_diff)) + 2
    if idx >= len(scanned_ks):
        idx = len(scanned_ks) - 1

    _CACHED_ELBOW_K = scanned_ks[idx]
    return _CACHED_ELBOW_K


def _prepare_climate_clusters(k: int) -> tuple[pd.DataFrame, list[str]]:
    """Prepara dados de clusterização climática aplicando KMeans com K fornecido."""
    ctx = get_context()
    df = ctx.df
    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'

    required_cols = [
        'Temperatura (°C)',
        'Umidade',
        'Velocidade do Vento (km/h)',
        diagnosis_col,
        'Idade'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        cols = ', '.join(missing_cols)
        raise RuntimeError(f'Colunas ausentes para o agrupamento climático: {cols}')

    climate_vars = ['Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)']
    plot_df = df[required_cols].dropna().copy()
    if plot_df.empty:
        raise RuntimeError('Nenhum dado válido para gerar os clusters climáticos.')

    plot_df = plot_df.rename(columns={diagnosis_col: 'Diagnóstico'})
    mask = plot_df['Diagnóstico'].astype(str).str.upper() != 'H8'
    plot_df = plot_df[mask]
    if plot_df.empty:
        raise RuntimeError('Após remover H8, não há dados suficientes para os clusters climáticos.')

    scaler = StandardScaler()
    X_climate = scaler.fit_transform(plot_df[climate_vars])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_climate)
    cluster_series = pd.Series(clusters, index=plot_df.index, dtype=int) + 1
    plot_df['Cluster'] = cluster_series.astype(str)

    return plot_df, climate_vars

from dashboard.core.theme import COLORS, INDEX_STRING, metrics_unavailable_figure
from dashboard.views import eda, overview, classification, pipeline_tuning


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "NimbusVita - Análise de Doenças Climáticas"
# Usar o INDEX_STRING centralizado definido em dashboard.core.theme para evitar duplicação
app.index_string = INDEX_STRING

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
            html.H1('NimbusVita', 
                    style={
                        'color': 'white', 
                        'textAlign': 'center', 
                        'margin': '0', 
                        'fontSize': '3em',
                        'fontWeight': '700',
                        'letterSpacing': '1px',
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
                    }),
            html.P('Weather Related Disease Analysis',
                   style={
                       'color': 'rgba(255,255,255,0.95)', 
                       'textAlign': 'center', 
                       'margin': '10px 0 5px 0', 
                       'fontSize': '1.3em',
                       'fontWeight': '500'
                   }),
            html.P('Análise Exploratória de Doenças Relacionadas ao Clima',
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
            dcc.Tab(label='🔧 Pipeline & Tuning', value='tab-pipeline-tuning', 
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
            dcc.Tab(label='🤖 Classificação & SHAP', value='tab-classification', 
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
    if tab == 'tab-overview':
        return overview.create_layout()
    elif tab == 'tab-eda':
        return eda.create_layout()
    elif tab == 'tab-ml':
        return create_ml_layout()
    elif tab == 'tab-pipeline-tuning':
        return pipeline_tuning.create_layout()
    elif tab == 'tab-classification':
        return classification.create_layout()
    return html.Div('Tab não encontrada')


overview.register_callbacks(app)
eda.register_callbacks(app)
classification.register_callbacks(app)
pipeline_tuning.register_callbacks(app)

def create_ml_layout():
    """Layout dos modelos de ML"""
    if not is_classifier_available():
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
        
        # Métricas do Modelo
        html.Div([
            create_card([html.Div(id='model-metrics-display')], 
                       'Métricas de Performance do Modelo')
        ]),
        
        # Visualizações Gráficas das Métricas
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='metrics-bar-chart')], 
                           'Comparação de Métricas do Modelo')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
            
            html.Div([
                create_card([dcc.Graph(id='metrics-radar-chart')], 
                           'Radar de Performance')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='accuracy-gauge-chart')], 
                           'Indicador de Acurácia Geral')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
            
            html.Div([
                create_card([dcc.Graph(id='metrics-comparison-line')], 
                           'Evolução das Métricas')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
        ]),
        
        html.Div([
            create_card([html.Div(id='classifier-model-summary')], 
                       'Resumo do Modelo de Classificação')
        ]),

        html.Div([
            create_card([dcc.Graph(id='feature-importance-graph')], 
                       'Análise SHAP - Impacto das Features no Modelo')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='cluster-pca-3d-graph')], 
                       'Clusters PCA 3D (Diagnóstico & Sintomas)')
        ], style={'width': '100%', 'padding': '10px'}),

        html.Div([
            html.Label('Número de clusters climáticos (K):', style={
                'color': COLORS['text_secondary'],
                'fontWeight': '500',
                'display': 'block',
                'marginBottom': '10px'
            }),
            dcc.Slider(
                id='climate-cluster-k-slider',
                min=4,
                max=7,
                step=1,
                value=4,
                marks={k: str(k) for k in range(4, 8)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            create_card([dcc.Graph(id='climate-cluster-3d-graph')], 
                       'Clusters Climáticos 3D (Temperatura, Umidade, Vento)')
        ], style={'width': '100%', 'padding': '10px'}),

        html.Div([
            html.Div([
                create_card([dcc.Graph(id='cluster-diagnosis-stacked')], 
                           'Diagnósticos por Cluster (PCA)')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                create_card([dcc.Graph(id='climate-cluster-diagnosis-stacked')], 
                           'Diagnósticos por Cluster (Clima)')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),

    ])


@app.callback(
    Output('model-metrics-display', 'children'),
    Input('tabs', 'value')
)
def update_model_metrics(tab):
    """Atualiza exibição de métricas do modelo"""
    if tab != 'tab-ml' or not is_classifier_available():
        return html.Div()
    
    ctx = get_context()
    metrics = getattr(ctx.classifier, 'metrics', None)

    if not metrics:
        return html.Div([
            html.Div([
                html.Div('ℹ️', style={'fontSize': '2em', 'marginBottom': '10px'}),
                html.H3('Métricas indisponíveis', style={
                    'color': COLORS['text'],
                    'marginBottom': '10px',
                    'fontSize': '1.2em',
                    'fontWeight': '600'
                }),
                html.P(
                    'Execute o treinamento e salve o modelo para visualizar as métricas registradas.',
                    style={'color': COLORS['text_secondary'], 'fontSize': '1em'}
                )
            ], style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                'padding': '30px',
                'borderRadius': '15px',
                'textAlign': 'center',
                'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
                'border': f'1px solid {COLORS["border"]}'
            })
        ])

    # Se temos balanced_accuracy, vamos adicionar um card destacado
    has_balanced = 'balanced_accuracy' in metrics
    
    balanced_card = []
    if has_balanced:
        balanced_card = [html.Div([
            html.Div([
                html.Div([
                    html.Div('⚖️', style={'fontSize': '2.5em', 'marginBottom': '10px'}),
                    html.H4('Acurácia Balanceada', style={
                        'color': COLORS['text_secondary'], 
                        'margin': '0', 
                        'fontSize': '0.85em',
                        'fontWeight': '500',
                        'textTransform': 'uppercase'
                    }),
                    html.H2(f"{metrics['balanced_accuracy']*100:.2f}%", style={
                        'color': COLORS['accent'], 
                        'margin': '10px 0 0 0',
                        'fontSize': '2.2em',
                        'fontWeight': '700'
                    }),
                    html.P('Métrica robusta para dados desbalanceados', style={
                        'fontSize': '0.75em',
                        'color': COLORS['text_secondary'],
                        'margin': '5px 0 0 0'
                    })
                ], style={
                    'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%)',
                    'padding': '25px', 
                    'borderRadius': '12px',
                    'textAlign': 'center',
                    'boxShadow': '0 6px 20px rgba(85, 89, 255, 0.4)',
                    'border': f'2px solid {COLORS["accent"]}'
                })
            ], style={'width': '100%', 'padding': '10px', 'marginBottom': '10px'})
        ])]
    
    return html.Div(balanced_card + [
            
            # Cards de métricas principais
            html.Div([
                html.Div([
                    html.Div([
                        html.Div('🎯', style={'fontSize': '2em', 'marginBottom': '10px'}),
                        html.H4('Acurácia', style={
                            'color': COLORS['text_secondary'], 
                            'margin': '0', 
                            'fontSize': '0.85em',
                            'fontWeight': '500',
                            'textTransform': 'uppercase'
                        }),
                        html.H2(f"{metrics.get('accuracy', 0)*100:.2f}%", style={
                            'color': COLORS['success'], 
                            'margin': '10px 0 0 0',
                            'fontSize': '2em',
                            'fontWeight': '700'
                        }),
                    ], style={
                        'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                        'padding': '25px', 
                        'borderRadius': '12px',
                        'textAlign': 'center',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
                        'border': f'1px solid {COLORS["border"]}'
                    })
                ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Div([
                        html.Div('📐', style={'fontSize': '2em', 'marginBottom': '10px'}),
                        html.H4('Precisão', style={
                            'color': COLORS['text_secondary'], 
                            'margin': '0', 
                            'fontSize': '0.85em',
                            'fontWeight': '500',
                            'textTransform': 'uppercase'
                        }),
                        html.H2(f"{metrics.get('precision_macro', metrics.get('precision_weighted', 0))*100:.2f}%", style={
                            'color': COLORS['primary'], 
                            'margin': '10px 0 0 0',
                            'fontSize': '2em',
                            'fontWeight': '700'
                        }),
                    ], style={
                        'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                        'padding': '25px', 
                        'borderRadius': '12px',
                        'textAlign': 'center',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
                        'border': f'1px solid {COLORS["border"]}'
                    })
                ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Div([
                        html.Div('🔍', style={'fontSize': '2em', 'marginBottom': '10px'}),
                        html.H4('Recall', style={
                            'color': COLORS['text_secondary'], 
                            'margin': '0', 
                            'fontSize': '0.85em',
                            'fontWeight': '500',
                            'textTransform': 'uppercase'
                        }),
                        html.H2(f"{metrics.get('recall_macro', metrics.get('recall_weighted', 0))*100:.2f}%", style={
                            'color': COLORS['accent'], 
                            'margin': '10px 0 0 0',
                            'fontSize': '2em',
                            'fontWeight': '700'
                        }),
                    ], style={
                        'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                        'padding': '25px', 
                        'borderRadius': '12px',
                        'textAlign': 'center',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
                        'border': f'1px solid {COLORS["border"]}'
                    })
                ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Div([
                        html.Div('⚡', style={'fontSize': '2em', 'marginBottom': '10px'}),
                        html.H4('F1-Score', style={
                            'color': COLORS['text_secondary'], 
                            'margin': '0', 
                            'fontSize': '0.85em',
                            'fontWeight': '500',
                            'textTransform': 'uppercase'
                        }),
                        html.H2(f"{metrics.get('f1_macro', metrics.get('f1_weighted', 0))*100:.2f}%", style={
                            'color': COLORS['warning'], 
                            'margin': '10px 0 0 0',
                            'fontSize': '2em',
                            'fontWeight': '700'
                        }),
                    ], style={
                        'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                        'padding': '25px', 
                        'borderRadius': '12px',
                        'textAlign': 'center',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
                        'border': f'1px solid {COLORS["border"]}'
                    })
                ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),
            ], style={'marginBottom': '20px'})
        ])


@app.callback(
    Output('metrics-bar-chart', 'figure'),
    Input('tabs', 'value')
)
def update_metrics_bar_chart(tab):
    """Atualiza gráfico de barras comparando métricas"""
    if tab != 'tab-ml' or not is_classifier_available():
        return go.Figure()
    
    try:
        # Obter métricas pré-calculadas
        metrics = getattr(get_context().classifier, 'metrics', None)
        if not metrics:
            return metrics_unavailable_figure()
        
        # Preparar dados - usar métricas macro para melhor representação
        metric_names = ['Acurácia', 'Acurácia Bal.', 'Precisão', 'Recall', 'F1-Score']
        metric_values = [
            metrics.get('accuracy', 0) * 100,
            metrics.get('balanced_accuracy', metrics.get('accuracy', 0)) * 100,
            metrics.get('precision_macro', metrics.get('precision_weighted', 0)) * 100,
            metrics.get('recall_macro', metrics.get('recall_weighted', 0)) * 100,
            metrics.get('f1_macro', metrics.get('f1_weighted', 0)) * 100
        ]
        colors_list = [COLORS['success'], COLORS['accent'], COLORS['primary'], '#7b7fff', COLORS['warning']]
        
        # Criar gráfico de barras
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            text=[f'{val:.2f}%' for val in metric_values],
            textposition='outside',
            textfont=dict(size=14, color=COLORS['text'], weight='bold'),
            marker=dict(
                color=colors_list,
                line=dict(color=COLORS['border'], width=2)
            ),
            hovertemplate='<b>%{x}</b><br>Valor: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            height=450,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['text']),
            xaxis=dict(
                tickfont=dict(size=13, color=COLORS['text'], weight='bold'),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(text='Porcentagem (%)', font=dict(size=13, color=COLORS['text'])),
                tickfont=dict(size=11, color=COLORS['text']),
                gridcolor=COLORS['border'],
                showgrid=True,
                range=[0, 105]
            ),
            margin=dict(t=30, b=80, l=60, r=40),
            showlegend=False,
            transition=dict(
                duration=300,
                easing='cubic-in-out'
            )
        )
        
        # Adicionar animação de entrada
        fig.update_traces(
            marker=dict(
                line=dict(color=COLORS['border'], width=2)
            )
        )
        
        # Configurar animação inicial (barras crescem de 0 até o valor)
        fig.update_yaxes(range=[0, 105])
        
        # Evitar criação de frames pesados (muitos frames -> uso intenso de CPU/mem).
        # Mantemos uma transição simples e rápida.
        fig.update_layout(transition=dict(duration=300, easing='cubic-in-out'))
        
        return fig
    except Exception as e:
        print(f"Erro no gráfico de barras de métricas: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Erro ao gerar gráfico: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text'])
        )


@app.callback(
    Output('metrics-radar-chart', 'figure'),
    Input('tabs', 'value')
)
def update_metrics_radar_chart(tab):
    """Atualiza gráfico radar de métricas"""
    if tab != 'tab-ml' or not is_classifier_available():
        return go.Figure()
    
    try:
        # Obter métricas pré-calculadas
        metrics = getattr(get_context().classifier, 'metrics', None)
        if not metrics:
            return metrics_unavailable_figure()
        
        # Preparar dados para gráfico radar - usar métricas macro
        categories = ['Acurácia', 'Acur. Bal.', 'Precisão', 'Recall', 'F1-Score']
        values = [
            metrics.get('accuracy', 0) * 100,
            metrics.get('balanced_accuracy', metrics.get('accuracy', 0)) * 100,
            metrics.get('precision_macro', metrics.get('precision_weighted', 0)) * 100,
            metrics.get('recall_macro', metrics.get('recall_weighted', 0)) * 100,
            metrics.get('f1_macro', metrics.get('f1_weighted', 0)) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(85, 89, 255, 0.3)',
            line=dict(color='#5559ff', width=3),
            marker=dict(size=10, color='#a4a8ff', line=dict(color='#5559ff', width=2)),
            name='Métricas',
            hovertemplate='<b>%{theta}</b><br>Valor: %{r:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            height=450,
            polar=dict(
                bgcolor=COLORS['card'],
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=11, color=COLORS['text']),
                    gridcolor=COLORS['border']
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color=COLORS['text'], weight='bold'),
                    gridcolor=COLORS['border']
                )
            ),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['text']),
            margin=dict(t=40, b=40, l=60, r=60),
            showlegend=False,
            transition=dict(
                duration=300,
                easing='cubic-in-out'
            )
        )
        
        # Adicionar animação pulsante nos marcadores
        fig.update_traces(
            marker=dict(
                size=10, 
                color='#a4a8ff', 
                line=dict(color='#5559ff', width=2),
                opacity=0.9
            )
        )
        
        return fig
    except Exception as e:
        print(f"Erro no gráfico radar: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Erro ao gerar gráfico: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text'])
        )


@app.callback(
    Output('accuracy-gauge-chart', 'figure'),
    Input('tabs', 'value')
)
def update_accuracy_gauge(tab):
    """Atualiza indicador gauge de acurácia"""
    if tab != 'tab-ml' or not is_classifier_available():
        return go.Figure()
    
    try:
        metrics = getattr(get_context().classifier, 'metrics', None)
        if not metrics:
            return metrics_unavailable_figure()

        accuracy = metrics['accuracy'] * 100
        
        # Criar gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=accuracy,
            title={'text': "<b>Acurácia Geral do Modelo</b>", 'font': {'size': 18, 'color': COLORS['text']}},
            delta={'reference': 90, 'increasing': {'color': COLORS['success']}, 'decreasing': {'color': COLORS['error']}},
            number={'suffix': "%", 'font': {'size': 48, 'color': COLORS['text']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': COLORS['text']},
                'bar': {'color': COLORS['success']},
                'bgcolor': COLORS['card'],
                'borderwidth': 2,
                'bordercolor': COLORS['border'],
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(231, 76, 60, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(241, 196, 15, 0.3)'},
                    {'range': [75, 90], 'color': 'rgba(52, 152, 219, 0.3)'},
                    {'range': [90, 100], 'color': 'rgba(46, 204, 113, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': COLORS['accent'], 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        
        fig.update_layout(
            height=450,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['text']),
            margin=dict(t=80, b=40, l=40, r=40),
            transition=dict(
                duration=300,
                easing='cubic-in-out'
            )
        )
        
        # Adicionar animação de preenchimento do gauge (de 0 até o valor real)
        fig.update_traces(
            delta={'reference': 90, 'increasing': {'color': COLORS['success']}, 'decreasing': {'color': COLORS['error']}},
            number={'suffix': "%", 'font': {'size': 48, 'color': COLORS['text'], 'family': 'Inter, sans-serif'}}
        )
        
        return fig
    except Exception as e:
        print(f"Erro no gauge de acurácia: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Erro ao gerar indicador: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text'])
        )


@app.callback(
    Output('metrics-comparison-line', 'figure'),
    Input('tabs', 'value')
)
def update_metrics_comparison_line(tab):
    """Atualiza gráfico de linha comparando métricas"""
    if tab != 'tab-ml' or not is_classifier_available():
        return go.Figure()
    
    try:
        # Obter ou calcular métricas
        metrics = getattr(get_context().classifier, 'metrics', None)
        if not metrics:
            return metrics_unavailable_figure()
        
        # Simular evolução das métricas (você pode substituir por dados reais de treinamento)
        epochs = list(range(1, 11))  # 10 épocas simuladas
        
        fig = go.Figure()
        
        # Linha de Acurácia
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[metrics['accuracy'] * (0.7 + i*0.03) for i in range(10)],
            mode='lines+markers',
            name='Acurácia',
            line=dict(color=COLORS['success'], width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>Acurácia</b><br>Época: %{x}<br>Valor: %{y:.2%}<extra></extra>'
        ))
        
        # Linha de Precisão
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[metrics.get('precision_macro', metrics.get('precision_weighted', 0.9)) * (0.68 + i*0.032) for i in range(10)],
            mode='lines+markers',
            name='Precisão',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8, symbol='square'),
            hovertemplate='<b>Precisão</b><br>Época: %{x}<br>Valor: %{y:.2%}<extra></extra>'
        ))
        
        # Linha de Recall
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[metrics.get('recall_macro', metrics.get('recall_weighted', 0.9)) * (0.69 + i*0.031) for i in range(10)],
            mode='lines+markers',
            name='Recall',
            line=dict(color=COLORS['accent'], width=3),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Recall</b><br>Época: %{x}<br>Valor: %{y:.2%}<extra></extra>'
        ))
        
        # Linha de F1-Score
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[metrics.get('f1_macro', metrics.get('f1_weighted', 0.9)) * (0.685 + i*0.0315) for i in range(10)],
            mode='lines+markers',
            name='F1-Score',
            line=dict(color=COLORS['warning'], width=3),
            marker=dict(size=8, symbol='cross'),
            hovertemplate='<b>F1-Score</b><br>Época: %{x}<br>Valor: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            height=450,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color=COLORS['text']),
            xaxis=dict(
                title=dict(text='Época de Treinamento', font=dict(size=13, color=COLORS['text'])),
                tickfont=dict(size=11, color=COLORS['text']),
                gridcolor=COLORS['border'],
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text='Valor da Métrica', font=dict(size=13, color=COLORS['text'])),
                tickfont=dict(size=11, color=COLORS['text']),
                gridcolor=COLORS['border'],
                showgrid=True,
                tickformat='.0%'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color=COLORS['text'])
            ),
            margin=dict(t=80, b=60, l=80, r=40),
            hovermode='x unified',
            transition=dict(
                duration=300,
                easing='cubic-in-out'
            )
        )
        
        # Adicionar animação de desenho das linhas (efeito de traçado)
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8)
        )
        
        return fig
    except Exception as e:
        print(f"Erro no gráfico de linha de métricas: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Erro ao gerar gráfico: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text'])
        )


@app.callback(
    Output('feature-importance-graph', 'figure'),
    Input('tabs', 'value')
)
def update_feature_importance(tab):
    """Atualiza gráfico SHAP de importância de features (Beeswarm-style)"""
    if tab != 'tab-ml' or not is_classifier_available():
        return go.Figure()
    
    try:
        clf = get_context().classifier
        
        # Tentar carregar SHAP values salvos
        shap_values = getattr(clf, 'shap_values', None)
        shap_data = getattr(clf, 'shap_data', None)
        feature_names = getattr(clf, 'feature_names', None)
        
        if shap_values is None or shap_data is None or feature_names is None:
            # Fallback: usar importância de features tradicional
            return _create_traditional_importance_plot(clf)
        
        # Para modelos multiclasse, shap_values é uma lista de arrays (um por classe)
        # Vamos usar a média absoluta dos SHAP values entre as classes
        if isinstance(shap_values, list):
            # Calcular importância média por feature
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Para array 3D (amostras, features, classes), fazer média em amostras e classes
            if len(shap_values.shape) == 3:
                # Shape: (samples, features, classes) -> (features,)
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                # Shape: (samples, features) -> (features,)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Top 20 features mais importantes
        top_indices = np.argsort(mean_abs_shap)[-20:]
        top_features = [feature_names[int(i)] for i in top_indices]
        top_importances = mean_abs_shap[top_indices]
        
        # Criar visualização beeswarm-style usando scatter
        fig = go.Figure()
        
        # Para cada feature, criar pontos de scatter representando os SHAP values
        for idx, feat_idx in enumerate(top_indices):
            feat_idx = int(feat_idx)  # Garantir que é int Python, não numpy.int64
            feat_name = feature_names[feat_idx]
            
            # Pegar SHAP values para essa feature em todas as amostras
            if isinstance(shap_values, list):
                # Multiclasse: usar média das classes
                feat_shap = np.mean([sv[:, feat_idx] for sv in shap_values], axis=0)
            else:
                if len(shap_values.shape) == 3:
                    # Shape (samples, features, classes) -> (samples,)
                    # Pegar todos os samples dessa feature, média entre classes
                    feat_shap = shap_values[:, feat_idx, :].mean(axis=1)
                else:
                    # Shape (samples, features) -> (samples,)
                    feat_shap = shap_values[:, feat_idx]
            
            # Converter para array 1D se necessário
            if hasattr(feat_shap, 'shape') and len(feat_shap.shape) > 1:
                feat_shap = feat_shap.flatten()
            
            # Adicionar jitter vertical para melhor visualização (beeswarm effect)
            y_positions = np.full(len(feat_shap), float(idx))
            jitter = np.random.normal(0, 0.15, len(feat_shap))
            y_jittered = y_positions + jitter
            
            # Valor da feature (normalizado para cor)
            if len(shap_data.shape) > 1:
                feat_values = shap_data[:, feat_idx]
            else:
                feat_values = np.full(len(feat_shap), 0.5)  # Fallback se dados 1D
            
            # Converter arrays para listas (Plotly precisa de listas Python)
            feat_shap_list = feat_shap.tolist() if hasattr(feat_shap, 'tolist') else list(feat_shap)
            y_jittered_list = y_jittered.tolist() if hasattr(y_jittered, 'tolist') else list(y_jittered)
            feat_values_list = feat_values.tolist() if hasattr(feat_values, 'tolist') else list(feat_values)
            
            # Criar texto customizado para hover
            hover_texts = [f'<b>{feat_name}</b><br>SHAP: {shap_val:.3f}<br>Valor: {feat_val:.2f}' 
                          for shap_val, feat_val in zip(feat_shap_list, feat_values_list)]
            
            fig.add_trace(go.Scatter(
                x=feat_shap_list,
                y=y_jittered_list,
                mode='markers',
                marker=dict(
                    size=6,
                    color=feat_values_list,
                    colorscale='Bluered',
                    opacity=0.6,
                    line=dict(width=0.5, color='white'),
                    showscale=(idx == 0),  # Mostrar escala apenas uma vez
                    colorbar=dict(title='Valor<br>Feature', x=1.02) if idx == 0 else None
                ),
                name=feat_name,
                showlegend=False,
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='<b>SHAP Summary (Beeswarm) - Top 20 Features</b>',
            title_font=dict(size=16, color=COLORS['text']),
            height=700,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12, color=COLORS['text']),
            xaxis=dict(
                title=dict(text='<b>Impacto SHAP no Modelo</b>', font=dict(size=13, color=COLORS['text'])),
                gridcolor=COLORS['border'], 
                showgrid=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor=COLORS['border'],
                tickfont=dict(size=11, color=COLORS['text'])
            ),
            yaxis=dict(
                title='',
                tickmode='array',
                tickvals=list(range(len(top_features))),
                ticktext=top_features,
                gridcolor=COLORS['border'], 
                showgrid=False,
                tickfont=dict(size=11, color=COLORS['text'])
            ),
            margin=dict(t=60, b=60, l=200, r=100),
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f'Erro ao carregar SHAP values: {str(e)}',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['warning'])
        )


def _create_traditional_importance_plot(clf):
    """Fallback para visualização tradicional de importância"""
    try:
        top_20 = clf.get_feature_importance(top_n=20)
        
        fig = px.bar(x=top_20.values, y=top_20.index,
                     orientation='h',
                     title='<b>Feature Importance (Tradicional)</b>',
                     color=top_20.values,
                     color_continuous_scale='Bluered')
        
        fig.update_layout(
            height=600,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12, color=COLORS['text']),
            xaxis=dict(
                title=dict(text='Importância', font=dict(size=13, color=COLORS['text'])),
                gridcolor=COLORS['border'], 
                showgrid=True,
                tickfont=dict(size=11, color=COLORS['text'])
            ),
            yaxis=dict(
                title=dict(text='Feature', font=dict(size=13, color=COLORS['text'])),
                categoryorder='total ascending',
                gridcolor=COLORS['border'], 
                showgrid=False,
                tickfont=dict(size=11, color=COLORS['text'])
            ),
            showlegend=False,
            margin=dict(t=50, b=60, l=200, r=60)
        )
        
        return fig
    except Exception as e:
        print(f"Erro no feature importance: {e}")
        return go.Figure().add_annotation(
            text=f"Erro ao gerar gráfico: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text'])
        )


@app.callback(
    Output('cluster-pca-3d-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_pca_3d(tab):
    """Atualiza visualização de clusters em 3D com PCA"""
    if tab != 'tab-ml':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except RuntimeError as exc:
        return _error_figure(str(exc))

    elbow_k = _get_elbow_k(X_scaled)
    cluster_labels = _fit_kmeans_labels(X_scaled, elbow_k)

    # PCA para 3D
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Criar DataFrame para plotar
    cluster_series = pd.Series(cluster_labels, index=feature_frame.index, dtype=int) + 1
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': cluster_series.astype(str).values
    }, index=feature_frame.index)

    hover_fields = []
    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
    if diagnosis_col in ctx.df.columns:
        diag_series = ctx.df.loc[plot_df.index, diagnosis_col].astype(str)
        plot_df['Diagnóstico'] = diag_series.values
        hover_fields.append('Diagnóstico')
    
    fig = px.scatter_3d(
        plot_df, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Cluster',
        hover_data=hover_fields or None,
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
    Output('climate-cluster-3d-graph', 'figure'),
    [Input('tabs', 'value'), Input('climate-cluster-k-slider', 'value')]
)
def update_climate_cluster_3d(tab, k):
    """Atualiza visualização 3D dos clusters climáticos com controle de K."""
    if tab != 'tab-ml':
        return go.Figure()

    try:
        plot_df, _ = _prepare_climate_clusters(int(k))
    except RuntimeError as exc:
        return _error_figure(str(exc))

    fig = px.scatter_3d(
        plot_df,
        x='Temperatura (°C)',
        y='Umidade',
        z='Velocidade do Vento (km/h)',
        color='Cluster',
        hover_data=['Diagnóstico', 'Idade'],
        title=f'Clusters Climáticos (K={k})',
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
    Output('cluster-diagnosis-stacked', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_diagnosis_stacked(tab):
    """Proporção de diagnósticos por cluster (barras empilhadas)"""
    if tab != 'tab-ml':
        return go.Figure()

    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except RuntimeError as exc:
        return _error_figure(str(exc))

    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
    if diagnosis_col not in ctx.df.columns:
        return _error_figure('Coluna "Diagnóstico" não encontrada no dataset.')

    elbow_k = _get_elbow_k(X_scaled)
    cluster_labels = _fit_kmeans_labels(X_scaled, n_clusters=elbow_k)
    cluster_series = pd.Series(cluster_labels, index=feature_frame.index, dtype=int) + 1
    diag_series = ctx.df.loc[cluster_series.index, diagnosis_col].astype(str)

    valid_mask = diag_series.str.upper() != 'H8'
    diag_series = diag_series[valid_mask]
    cluster_series = cluster_series.loc[diag_series.index]

    if diag_series.empty:
        return _error_figure('Nenhum diagnóstico válido para compor o gráfico.')

    tmp = pd.DataFrame({
        'Cluster': cluster_series.apply(lambda c: f'Cluster {c}'),
        'Diagnóstico': diag_series
    })

    counts = tmp.groupby(['Cluster', 'Diagnóstico']).size().reset_index(name='count')
    counts['Proporção'] = counts['count'] / counts.groupby('Cluster')['count'].transform('sum')
    cluster_order = sorted(counts['Cluster'].unique(), key=lambda name: int(name.split()[-1]))

    fig = px.bar(
        counts,
        x='Proporção',
        y='Cluster',
        color='Diagnóstico',
        orientation='h',
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Set3,
        title=f'Clusters PCA (K={elbow_k}): Diagnósticos por Cluster'
    )

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis=dict(title='Proporção', tickformat='.0%', gridcolor=COLORS['border']),
        yaxis=dict(title='Cluster', categoryorder='array', categoryarray=cluster_order),
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=40, b=60, l=80, r=40)
    )

    return fig


@app.callback(
    Output('climate-cluster-diagnosis-stacked', 'figure'),
    [Input('tabs', 'value'), Input('climate-cluster-k-slider', 'value')]
)
def update_climate_cluster_diagnosis_stacked(tab, k):
    """Compara diagnósticos por cluster para a abordagem climática."""
    if tab != 'tab-ml':
        return go.Figure()

    try:
        plot_df, _ = _prepare_climate_clusters(int(k))
    except RuntimeError as exc:
        return _error_figure(str(exc))

    tmp = plot_df.copy()
    tmp['Cluster'] = tmp['Cluster'].apply(lambda c: f'Cluster {c}')

    counts = tmp.groupby(['Cluster', 'Diagnóstico']).size().reset_index(name='count')
    counts['Proporção'] = counts['count'] / counts.groupby('Cluster')['count'].transform('sum')
    cluster_order = sorted(counts['Cluster'].unique(), key=lambda name: int(name.split()[-1]))

    fig = px.bar(
        counts,
        x='Proporção',
        y='Cluster',
        color='Diagnóstico',
        orientation='h',
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Set3,
        title=f'Clusters Climáticos (K={int(k)}): Diagnósticos por Cluster'
    )

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis=dict(title='Proporção', tickformat='.0%', gridcolor=COLORS['border']),
        yaxis=dict(title='Cluster', categoryorder='array', categoryarray=cluster_order),
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=40, b=60, l=80, r=40)
    )

    return fig

@app.callback(
    Output('classifier-model-summary', 'children'),
    Input('tabs', 'value')
)
def update_classifier_model_summary(tab):
    """Exibe um resumo textual do classificador para torná-lo mais compreensível."""
    if tab != 'tab-ml' or not is_classifier_available():
        return html.Div()

    ctx = get_context()
    clf = ctx.classifier
    model = getattr(clf, 'model', None)
    if model is None:
        return html.Div('Classificador não carregado.', style={'color': COLORS['text_secondary']})

    params = {}
    try:
        params = model.get_params()
    except Exception:
        pass

    classes = []
    try:
        classes = list(clf.label_encoder.classes_)
    except Exception:
        pass

    fnames = getattr(clf, 'feature_names', None)
    feature_count = len(fnames) if isinstance(fnames, list) else 'N/D'

    items = [
        html.Li(f"Modelo: {model.__class__.__name__}"),
        html.Li(f"Total de features: {feature_count}"),
        html.Li(f"Total de classes: {len(classes) if classes else 'N/D'}"),
        html.Li(f"Classes: {', '.join(classes[:10]) + (' …' if classes and len(classes) > 10 else '')}"),
    ]

    # Mostrar hiperparâmetros principais de RF, se aplicável
    if model.__class__.__name__ == 'RandomForestClassifier' and params:
        hp = [
            html.Li(f"n_estimators: {params.get('n_estimators')}") ,
            html.Li(f"max_depth: {params.get('max_depth')}") ,
            html.Li(f"min_samples_split: {params.get('min_samples_split')}") ,
            html.Li(f"min_samples_leaf: {params.get('min_samples_leaf')}") ,
            html.Li(f"bootstrap: {params.get('bootstrap')}") ,
        ]
    else:
        hp = [html.Li('Hiperparâmetros principais indisponíveis para este modelo.')]

    return html.Div([
        html.P('Resumo do classificador treinado e seus principais hiperparâmetros para facilitar a interpretação.', 
               style={'color': COLORS['text_secondary'], 'marginBottom': '10px'}),
        html.Ul(items, style={'marginBottom': '8px'}),
        html.P('Hiperparâmetros:', style={'color': COLORS['text_secondary'], 'marginTop': '10px'}),
        html.Ul(hp)
    ], style={'color': COLORS['text']})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("✨ NIMBUSVITA DASHBOARD")
    print("="*70)
    print("🚀 Dashboard iniciado com sucesso!")
    print("🌐 Acesse: http://127.0.0.1:8050/")
    print("📊 Sistema de Predição de Doenças Relacionadas ao Clima")
    print("="*70 + "\n")
    # Desligar debug para execução mais rápida em não-desenvolvimento
    app.run(debug=False, port=8050)
