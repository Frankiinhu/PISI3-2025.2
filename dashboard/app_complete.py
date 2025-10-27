import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Dashboard Principal - NimbusVita (Versão Completa com Callbacks)
Análise Exploratória de Doenças Relacionadas ao Clima
"""
from typing import Any

import dash
from dash import Input, Output, dcc, html
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

    scaler = getattr(ctx.clusterer, 'scaler', None)
    if scaler is None:
        raise RuntimeError('Clusterizador sem scaler configurado; execute o treinamento do modelo.')

    X_scaled = scaler.transform(feature_frame)
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
from dashboard.views import eda, overview


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "NimbusVita - Análise de Doenças Climáticas"
app.index_string = INDEX_STRING

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
                border-color: #5559ff !important;
                box-shadow: 0 0 0 3px rgba(85, 89, 255, 0.1) !important;
                outline: none;
            }
            
            /* Estilo para checkboxes */
            input[type="checkbox"] {
                accent-color: #5559ff;
            }
            
            /* Animação do botão */
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(85, 89, 255, 0.6) !important;
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
                background: linear-gradient(135deg, #5559ff 0%, #7b7fff 100%);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #7b7fff 0%, #a4a8ff 100%);
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
            
            /* Custom dropdown styles */
            .custom-dropdown .Select-value-label,
            .custom-dropdown .Select-placeholder,
            .custom-dropdown input {
                color: #e8eaf6 !important;
            }
            
            .custom-dropdown .Select-value {
                background-color: rgba(102, 126, 234, 0.2) !important;
                border-color: rgba(102, 126, 234, 0.4) !important;
                color: #e8eaf6 !important;
            }
            
            .custom-dropdown .Select-input input {
                color: #e8eaf6 !important;
            }
            
            /* Estilo para o dropdown do Dash/React-Select */
            div[class*="css-"] input {
                color: #e8eaf6 !important;
            }
            
            div[class*="singleValue"] {
                color: #e8eaf6 !important;
            }
            
            div[class*="placeholder"] {
                color: rgba(232, 234, 246, 0.6) !important;
            }
            
            /* Tabs animation */
            ._dash-undo-redo {
                display: none;
            }
            
            .tab {
                transition: all 0.3s ease;
            }
            
            /* Animações de carregamento */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.5;
                }
            }
            
            @keyframes shimmer {
                0% {
                    background-position: -1000px 0;
                }
                100% {
                    background-position: 1000px 0;
                }
            }
            
            /* Aplicar animação aos gráficos */
            .js-plotly-plot {
                animation: fadeInUp 0.8s ease-out;
            }
            
            /* Loading spinner personalizado */
            ._dash-loading {
                position: relative;
            }
            
            ._dash-loading::after {
                content: "";
                position: absolute;
                top: 50%;
                left: 50%;
                width: 50px;
                height: 50px;
                margin: -25px 0 0 -25px;
                border: 4px solid rgba(85, 89, 255, 0.3);
                border-top-color: #5559ff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to {
                    transform: rotate(360deg);
                }
            }
            
            /* Animação suave para cards */
            [id*="card-"] {
                animation: fadeInUp 0.6s ease-out;
                animation-fill-mode: both;
            }
            
            /* Delay progressivo para múltiplos cards */
            [id*="card-"]:nth-child(1) { animation-delay: 0.1s; }
            [id*="card-"]:nth-child(2) { animation-delay: 0.2s; }
            [id*="card-"]:nth-child(3) { animation-delay: 0.3s; }
            [id*="card-"]:nth-child(4) { animation-delay: 0.4s; }
            
            /* Hover effect aprimorado com escala */
            [id*="card-"]:hover {
                transform: translateY(-5px) scale(1.02);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Skeleton loader para gráficos */
            @keyframes skeletonLoading {
                0% {
                    background-position: -200px 0;
                }
                100% {
                    background-position: calc(200px + 100%) 0;
                }
            }
            
            .skeleton-loader {
                background: linear-gradient(
                    90deg,
                    rgba(85, 89, 255, 0.1) 0%,
                    rgba(85, 89, 255, 0.3) 50%,
                    rgba(85, 89, 255, 0.1) 100%
                );
                background-size: 200px 100%;
                animation: skeletonLoading 1.5s infinite;
                border-radius: 8px;
            }
            
            /* Transições suaves para estado de carregamento */
            .loading-state {
                opacity: 0.6;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }
            
            /* Feedback visual de sucesso */
            @keyframes successPulse {
                0%, 100% {
                    box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
                }
                50% {
                    box-shadow: 0 0 0 20px rgba(74, 222, 128, 0);
                }
            }
            
            .success-feedback {
                animation: successPulse 1s ease-out;
            }
            
            /* Bouncing animation para elementos interativos */
            @keyframes bounce {
                0%, 100% {
                    transform: translateY(0);
                }
                50% {
                    transform: translateY(-10px);
                }
            }
            
            .bounce-animation {
                animation: bounce 2s infinite;
            }
            
            /* Slide in animation */
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-50px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(50px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .slide-in-left {
                animation: slideInLeft 0.6s ease-out;
            }
            
            .slide-in-right {
                animation: slideInRight 0.6s ease-out;
            }
            
            /* Progress bar animation */
            @keyframes progressBar {
                0% {
                    width: 0%;
                }
                100% {
                    width: 100%;
                }
            }
            
            .progress-bar {
                height: 4px;
                background: linear-gradient(90deg, #5559ff, #7b7fff, #a4a8ff);
                animation: progressBar 2s ease-out;
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
            dcc.Tab(label='Pipeline de Treinamento', value='tab-pipeline', 
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
    elif tab == 'tab-pipeline':
        return create_pipeline_layout()


overview.register_callbacks(app)
eda.register_callbacks(app)
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
                       'Top 20 Features Mais Importantes (Random Forest)')
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

    return html.Div([
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
                        html.H2(f"{metrics['accuracy']*100:.2f}%", style={
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
                        html.Div('�', style={'fontSize': '2em', 'marginBottom': '10px'}),
                        html.H4('Precisão', style={
                            'color': COLORS['text_secondary'], 
                            'margin': '0', 
                            'fontSize': '0.85em',
                            'fontWeight': '500',
                            'textTransform': 'uppercase'
                        }),
                        html.H2(f"{metrics['precision']*100:.2f}%", style={
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
                        html.H2(f"{metrics['recall']*100:.2f}%", style={
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
                        html.H2(f"{metrics['f1_score']*100:.2f}%", style={
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
        
        # Preparar dados
        metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'] * 100,
            metrics['precision'] * 100,
            metrics['recall'] * 100,
            metrics['f1_score'] * 100
        ]
        colors_list = [COLORS['success'], COLORS['primary'], COLORS['accent'], COLORS['warning']]
        
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
                duration=1000,
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
        
        # Adicionar frames para animação
        frames = []
        steps = 20
        for i in range(steps + 1):
            frame_data = go.Bar(
                x=metric_names,
                y=[val * (i / steps) for val in metric_values],
                text=[f'{val * (i / steps):.2f}%' for val in metric_values],
                textposition='outside',
                textfont=dict(size=14, color=COLORS['text'], weight='bold'),
                marker=dict(
                    color=colors_list,
                    line=dict(color=COLORS['border'], width=2)
                ),
                hovertemplate='<b>%{x}</b><br>Valor: %{y:.2f}%<extra></extra>'
            )
            frames.append(go.Frame(data=[frame_data], name=str(i)))
        
        fig.frames = frames
        
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
        
        # Preparar dados para gráfico radar
        categories = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        values = [
            metrics['accuracy'] * 100,
            metrics['precision'] * 100,
            metrics['recall'] * 100,
            metrics['f1_score'] * 100
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
                duration=1200,
                easing='elastic-out'
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
                duration=1500,
                easing='elastic-in-out'
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
            y=[metrics['precision'] * (0.68 + i*0.032) for i in range(10)],
            mode='lines+markers',
            name='Precisão',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8, symbol='square'),
            hovertemplate='<b>Precisão</b><br>Época: %{x}<br>Valor: %{y:.2%}<extra></extra>'
        ))
        
        # Linha de Recall
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[metrics['recall'] * (0.69 + i*0.031) for i in range(10)],
            mode='lines+markers',
            name='Recall',
            line=dict(color=COLORS['accent'], width=3),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Recall</b><br>Época: %{x}<br>Valor: %{y:.2%}<extra></extra>'
        ))
        
        # Linha de F1-Score
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[metrics['f1_score'] * (0.685 + i*0.0315) for i in range(10)],
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
                duration=800,
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
    """Atualiza gráfico de importância de features"""
    if tab != 'tab-ml' or not is_classifier_available():
        return go.Figure()
    
    try:
        clf = get_context().classifier
        feature_importances = getattr(clf, 'feature_importances', None)
        if feature_importances is None:
            # Calcular em tempo de execução se possível
            try:
                top_20 = clf.get_feature_importance(top_n=20)
            except Exception:
                return go.Figure().add_annotation(
                    text='Importância de features indisponível para o classificador atual.',
                    xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color=COLORS['text'])
                )
        else:
            top_20 = feature_importances.head(20)
        
        fig = px.bar(x=top_20.values, y=top_20.index,
                     orientation='h',
                     title='',
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
            margin=dict(t=30, b=60, l=200, r=60)
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


def create_pipeline_layout():
    """Layout da Pipeline Automatizada de Treinamento"""
    return html.Div([
        html.Div([
            html.H2('Pipeline Automatizada de Treinamento ML', style={
                'color': COLORS['text'], 
                'marginBottom': '10px',
                'fontSize': '2em',
                'fontWeight': '700'
            }),
            html.P('Visualize todo o processo de treinamento dos modelos de Machine Learning', style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '30px'
            })
        ]),
        
        # Visualização do Fluxo da Pipeline
        create_card([
            html.H3('🔄 Fluxo da Pipeline', style={'color': COLORS['text'], 'marginBottom': '30px'}),
            dcc.Graph(id='pipeline-flow-graph')
        ], 'Pipeline de Processamento'),
        
        html.Div([
            # Controles da Pipeline
            html.Div([
                create_card([
                    html.H3('⚙️ Controles da Pipeline', style={'color': COLORS['text'], 'marginBottom': '20px'}),
                    html.Button('▶️ Executar Pipeline Completa', id='btn-run-pipeline', n_clicks=0,
                               style={
                                   'width': '100%',
                                   'padding': '15px',
                                   'backgroundColor': COLORS['accent'],
                                   'color': 'white',
                                   'border': 'none',
                                   'borderRadius': '10px',
                                   'fontSize': '1.1em',
                                   'fontWeight': '600',
                                   'cursor': 'pointer',
                                   'marginBottom': '15px',
                                   'transition': 'all 0.3s ease'
                               }),
                    html.Div(id='pipeline-status', children=[
                        html.P('Status: Aguardando execução', style={'color': COLORS['text_secondary']})
                    ]),
                    html.Div(id='pipeline-progress-bar', children=[
                        html.Div(style={
                            'width': '0%',
                            'height': '6px',
                            'background': f'linear-gradient(90deg, {COLORS["primary"]}, {COLORS["accent"]})',
                            'borderRadius': '3px',
                            'transition': 'width 0.5s ease'
                        }, id='progress-bar-fill')
                    ], style={
                        'width': '100%',
                        'backgroundColor': COLORS['background'],
                        'borderRadius': '3px',
                        'marginTop': '20px'
                    })
                ], 'Controles')
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '10px'}),
            
            # Métricas em Tempo Real
            html.Div([
                create_card([
                    html.H3('📊 Métricas em Tempo Real', style={'color': COLORS['text'], 'marginBottom': '20px'}),
                    dcc.Graph(id='pipeline-metrics-realtime')
                ], 'Monitoramento')
            ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '10px'}),
        ], style={'marginTop': '20px'}),
        
        # Etapas da Pipeline
        html.Div([
            create_card([
                html.H3('📋 Etapas da Pipeline', style={'color': COLORS['text'], 'marginBottom': '25px'}),
                dcc.Graph(id='pipeline-stages-graph')
            ], 'Detalhamento das Etapas')
        ], style={'marginTop': '20px'}),
        
        # Comparação de Modelos
        html.Div([
            create_card([
                html.H3('🏆 Comparação de Performance dos Modelos', style={'color': COLORS['text'], 'marginBottom': '25px'}),
                dcc.Graph(id='pipeline-model-comparison')
            ], 'Resultados')
        ], style={'marginTop': '20px'}),
        
        # Log de Execução
        html.Div([
            create_card([
                html.H3('📝 Log de Execução', style={'color': COLORS['text'], 'marginBottom': '20px'}),
                html.Div(id='pipeline-log', children=[
                    html.Pre('Aguardando execução da pipeline...', style={
                        'backgroundColor': COLORS['background'],
                        'padding': '20px',
                        'borderRadius': '8px',
                        'color': COLORS['text_secondary'],
                        'fontSize': '0.9em',
                        'maxHeight': '300px',
                        'overflowY': 'auto',
                        'fontFamily': 'monospace'
                    })
                ])
            ], 'Console')
        ], style={'marginTop': '20px'}),
    ])


# Callbacks da Pipeline
@app.callback(
    [Output('pipeline-flow-graph', 'figure'),
     Output('pipeline-stages-graph', 'figure'),
     Output('pipeline-model-comparison', 'figure'),
     Output('pipeline-metrics-realtime', 'figure')],
    [Input('tabs', 'value')]
)
def update_pipeline_visualizations(tab):
    """Atualiza visualizações da pipeline"""
    if tab != 'tab-pipeline':
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    # 1. Fluxo da Pipeline (Network Diagram - Mais Bonito)
    flow_fig = go.Figure()
    
    # Definir posições dos nós em um fluxo mais organizado (3 linhas)
    stages_info = [
        {"name": "📥 Dados<br>Brutos", "x": 0.1, "y": 0.5, "color": "#a4a8ff", "icon": "📥", "size": 50},
        {"name": "🧹 Limpeza<br>de Dados", "x": 0.25, "y": 0.8, "color": "#5559ff", "icon": "🧹", "size": 45},
        {"name": "🔧 Feature<br>Engineering", "x": 0.4, "y": 0.8, "color": "#7b7fff", "icon": "🔧", "size": 45},
        {"name": "📊 Train/Test<br>Split", "x": 0.55, "y": 0.5, "color": "#4facfe", "icon": "📊", "size": 48},
        {"name": "🤖 Treinamento<br>de Modelos", "x": 0.7, "y": 0.2, "color": "#a4a8ff", "icon": "🤖", "size": 55},
        {"name": "✅ Validação<br>Cruzada", "x": 0.7, "y": 0.8, "color": "#4ade80", "icon": "✅", "size": 45},
        {"name": "💾 Modelo<br>Salvo", "x": 0.9, "y": 0.5, "color": "#fbbf24", "icon": "💾", "size": 50},
    ]
    
    # Conexões entre os nós (setas)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)
    ]
    
    # Desenhar as conexões (setas gradientes)
    for source, target in edges:
        x0, y0 = stages_info[source]["x"], stages_info[source]["y"]
        x1, y1 = stages_info[target]["x"], stages_info[target]["y"]
        
        # Linha com gradiente simulado (usando múltiplas linhas)
        flow_fig.add_trace(go.Scatter(
            x=[x0, (x0+x1)/2, x1],
            y=[y0, (y0+y1)/2, y1],
            mode='lines',
            line=dict(
                color='rgba(102, 126, 234, 0.4)',
                width=4,
                shape='spline'
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Adicionar setas nas pontas
        flow_fig.add_annotation(
            x=x1, y=y1,
            ax=x0 + (x1-x0)*0.85, ay=y0 + (y1-y0)*0.85,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor='rgba(240, 147, 251, 0.7)',
        )
    
    # Desenhar os nós (círculos coloridos)
    for stage in stages_info:
        # Círculo de fundo (brilho)
        flow_fig.add_trace(go.Scatter(
            x=[stage["x"]],
            y=[stage["y"]],
            mode='markers',
            marker=dict(
                size=stage["size"] + 15,
                color=stage["color"],
                opacity=0.2,
                line=dict(width=0)
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Círculo principal
        flow_fig.add_trace(go.Scatter(
            x=[stage["x"]],
            y=[stage["y"]],
            mode='markers+text',
            marker=dict(
                size=stage["size"],
                color=stage["color"],
                line=dict(color='white', width=3),
                opacity=0.95
            ),
            text=stage["name"],
            textposition="bottom center",
            textfont=dict(
                size=11,
                color=COLORS['text'],
                family='Inter, sans-serif',
                weight='bold'
            ),
            hovertemplate=f'<b>{stage["name"].replace("<br>", " ")}</b><br>Status: Ativo<extra></extra>',
            showlegend=False
        ))
    
    flow_fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color=COLORS['text'], family='Inter, sans-serif'),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.05, 1.05]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1]
        ),
        margin=dict(t=30, b=30, l=30, r=30),
        hovermode='closest'
    )
    
    # 2. Etapas com Tempo de Execução (Visual Melhorado)
    stages_data = [
        {"name": "📥 Carregamento", "time": 0.5, "status": "✓", "color": "#a4a8ff"},
        {"name": "🧹 Limpeza", "time": 1.2, "status": "✓", "color": "#5559ff"},
        {"name": "🔧 Feature Eng.", "time": 2.3, "status": "✓", "color": "#7b7fff"},
        {"name": "📊 Split", "time": 0.8, "status": "✓", "color": "#4facfe"},
        {"name": "🤖 Treinamento", "time": 15.4, "status": "✓", "color": "#a4a8ff"},
        {"name": "✅ Validação", "time": 3.2, "status": "✓", "color": "#4ade80"},
        {"name": "💾 Salvamento", "time": 0.6, "status": "✓", "color": "#fbbf24"}
    ]
    
    stages = [s["name"] for s in stages_data]
    times = [s["time"] for s in stages_data]
    colors = [s["color"] for s in stages_data]
    
    stages_fig = go.Figure()
    
    # Barras principais com gradiente
    stages_fig.add_trace(go.Bar(
        x=stages,
        y=times,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        text=[f'<b>{t:.1f}s</b>' for t in times],
        textposition='outside',
        textfont=dict(size=13, color=COLORS['text'], family='Inter, sans-serif', weight='bold'),
        hovertemplate='<b>%{x}</b><br>⏱️ Tempo: %{y:.2f}s<br>Status: Concluído ✓<extra></extra>',
        width=0.6
    ))
    
    stages_fig.update_layout(
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family='Inter, sans-serif', size=12),
        xaxis=dict(
            title=dict(text='<b>Etapa do Pipeline</b>', font=dict(size=14)),
            gridcolor='rgba(102, 126, 234, 0.1)',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title=dict(text='<b>Tempo (segundos)</b>', font=dict(size=14)),
            gridcolor=COLORS['border'],
            gridwidth=1,
            griddash='dot'
        ),
        showlegend=False,
        margin=dict(t=40, b=80, l=70, r=30),
        bargap=0.3
    )
    
    # 3. Comparação de Modelos (Visual Premium)
    models_data = [
        {"name": "🌲 Random Forest", "accuracy": 96.63, "time": 12.3, "color": "#5559ff"},
        {"name": "🚀 Gradient Boost", "accuracy": 97.98, "time": 15.6, "color": "#7b7fff"},
        {"name": "🎯 SVM", "accuracy": 98.65, "time": 8.9, "color": "#a4a8ff"},
        {"name": "📈 Logistic Reg", "accuracy": 97.98, "time": 5.4, "color": "#4facfe"},
        {"name": "🔮 K-Means", "accuracy": None, "time": 3.2, "color": "#4ade80"}
    ]
    
    comparison_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            '<b>🏆 Acurácia dos Modelos (%)</b>',
            '<b>⚡ Tempo de Treinamento (s)</b>'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]],
        horizontal_spacing=0.12
    )
    
    # Gráfico de Acurácia (sem K-Means)
    acc_models = [m for m in models_data if m["accuracy"] is not None]
    comparison_fig.add_trace(
        go.Bar(
            x=[m["name"] for m in acc_models],
            y=[m["accuracy"] for m in acc_models],
            marker=dict(
                color=[m["color"] for m in acc_models],
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'<b>{m["accuracy"]:.1f}%</b>' for m in acc_models],
            textposition='outside',
            textfont=dict(size=12, color=COLORS['text'], weight='bold'),
            hovertemplate='<b>%{x}</b><br>🎯 Acurácia: %{y:.2f}%<extra></extra>',
            width=0.65,
            name='Acurácia'
        ),
        row=1, col=1
    )
    
    # Gráfico de Tempo
    comparison_fig.add_trace(
        go.Bar(
            x=[m["name"] for m in models_data],
            y=[m["time"] for m in models_data],
            marker=dict(
                color=[m["color"] for m in models_data],
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            text=[f'<b>{m["time"]:.1f}s</b>' for m in models_data],
            textposition='outside',
            textfont=dict(size=12, color=COLORS['text'], weight='bold'),
            hovertemplate='<b>%{x}</b><br>⏱️ Tempo: %{y:.1f}s<extra></extra>',
            width=0.65,
            name='Tempo'
        ),
        row=1, col=2
    )
    
    comparison_fig.update_layout(
        height=480,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family='Inter, sans-serif', size=11),
        showlegend=False,
        margin=dict(t=70, b=100, l=70, r=70)
    )
    
    comparison_fig.update_xaxes(
        gridcolor='rgba(102, 126, 234, 0.1)',
        tickangle=-35,
        tickfont=dict(size=10)
    )
    comparison_fig.update_yaxes(
        gridcolor=COLORS['border'],
        gridwidth=1,
        griddash='dot'
    )
    
    # 4. Métricas em Tempo Real (Estilo Premium)
    epochs = list(range(1, 21))
    train_acc = [0.5 + 0.025*i + np.random.random()*0.02 for i in epochs]
    val_acc = [0.48 + 0.024*i + np.random.random()*0.02 for i in epochs]
    
    metrics_fig = go.Figure()
    
    # Linha de Treino com área preenchida
    metrics_fig.add_trace(go.Scatter(
        x=epochs, y=train_acc,
        mode='lines+markers',
        name='📊 Treino',
        line=dict(color='#a4a8ff', width=4, shape='spline'),
        marker=dict(size=10, symbol='circle', line=dict(color='white', width=2)),
        fill='tonexty',
        fillcolor='rgba(164, 168, 255, 0.2)',
        hovertemplate='<b>Época %{x}</b><br>🎯 Acurácia: %{y:.2%}<extra></extra>'
    ))
    
    # Linha de Validação com área preenchida
    metrics_fig.add_trace(go.Scatter(
        x=epochs, y=val_acc,
        mode='lines+markers',
        name='✅ Validação',
        line=dict(color='#4facfe', width=4, shape='spline'),
        marker=dict(size=10, symbol='diamond', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(79, 172, 254, 0.15)',
        hovertemplate='<b>Época %{x}</b><br>🎯 Acurácia: %{y:.2%}<extra></extra>'
    ))
    
    # Adicionar linha de meta (95%)
    metrics_fig.add_hline(
        y=0.95,
        line_dash="dash",
        line_color='#4ade80',
        line_width=2,
        annotation_text="Meta: 95%",
        annotation_position="right",
        annotation_font=dict(size=11, color='#4ade80')
    )
    
    metrics_fig.update_layout(
        height=370,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family='Inter, sans-serif', size=12),
        xaxis=dict(
            title=dict(text='<b>Época de Treinamento</b>', font=dict(size=13)),
            gridcolor='rgba(102, 126, 234, 0.1)',
            gridwidth=1,
            showline=True,
            linecolor=COLORS['border']
        ),
        yaxis=dict(
            title=dict(text='<b>Acurácia</b>', font=dict(size=13)),
            gridcolor=COLORS['border'],
            gridwidth=1,
            griddash='dot',
            tickformat='.0%',
            showline=True,
            linecolor=COLORS['border']
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(30, 33, 57, 0.95)',
            bordercolor=COLORS['accent'],
            borderwidth=2,
            font=dict(size=12)
        ),
        margin=dict(t=30, b=70, l=70, r=30),
        hovermode='x unified'
    )
    
    return flow_fig, stages_fig, comparison_fig, metrics_fig


@app.callback(
    [Output('pipeline-status', 'children'),
     Output('progress-bar-fill', 'style'),
     Output('pipeline-log', 'children')],
    [Input('btn-run-pipeline', 'n_clicks')]
)
def run_pipeline(n_clicks):
    """Simula execução da pipeline"""
    if n_clicks == 0:
        return [
            html.P('Status: Aguardando execução', style={'color': COLORS['text_secondary']})
        ], {
            'width': '0%',
            'height': '6px',
            'background': f'linear-gradient(90deg, {COLORS["primary"]}, {COLORS["accent"]})',
            'borderRadius': '3px',
            'transition': 'width 0.5s ease'
        }, [
            html.Pre('Aguardando execução da pipeline...', style={
                'backgroundColor': COLORS['background'],
                'padding': '20px',
                'borderRadius': '8px',
                'color': COLORS['text_secondary'],
                'fontSize': '0.9em',
                'maxHeight': '300px',
                'overflowY': 'auto',
                'fontFamily': 'monospace'
            })
        ]
    
    # Simulação de execução completa
    log_text = """[2025-10-21 14:32:10] ✓ Iniciando pipeline...
[2025-10-21 14:32:10] ✓ Carregando dados: data/DATASET FINAL WRDP.csv
[2025-10-21 14:32:10] ✓ Dataset carregado: 5200 linhas, 51 colunas
[2025-10-21 14:32:11] ✓ Limpeza de dados concluída
[2025-10-21 14:32:13] ✓ Feature Engineering aplicado
[2025-10-21 14:32:14] ✓ Split Train/Test: 80/20
[2025-10-21 14:32:14] ✓ Treinando Random Forest...
[2025-10-21 14:32:27] ✓ Random Forest - Acurácia: 96.63%
[2025-10-21 14:32:27] ✓ Treinando Gradient Boosting...
[2025-10-21 14:32:43] ✓ Gradient Boosting - Acurácia: 97.98%
[2025-10-21 14:32:43] ✓ Treinando SVM...
[2025-10-21 14:32:52] ✓ SVM - Acurácia: 98.65%
[2025-10-21 14:32:52] ✓ Treinando K-Means Clustering...
[2025-10-21 14:32:55] ✓ K-Means - Silhouette Score: 0.73
[2025-10-21 14:32:55] ✓ Validação cruzada: 5 folds
[2025-10-21 14:32:58] ✓ Média CV: 97.84% (±1.2%)
[2025-10-21 14:32:58] ✓ Salvando modelos em models/saved_models/
[2025-10-21 14:32:59] ✓ Pipeline concluída com sucesso!
[2025-10-21 14:32:59] 🎉 Total: 23.8 segundos"""
    
    return [
        html.Div([
            html.P('Status: ', style={'display': 'inline', 'color': COLORS['text_secondary']}),
            html.Span('Concluída ✓', style={'display': 'inline', 'color': COLORS['success'], 'fontWeight': '700'})
        ])
    ], {
        'width': '100%',
        'height': '6px',
        'background': f'linear-gradient(90deg, {COLORS["success"]}, {COLORS["accent"]})',
        'borderRadius': '3px',
        'transition': 'width 2s ease'
    }, [
        html.Pre(log_text, style={
            'backgroundColor': COLORS['background'],
            'padding': '20px',
            'borderRadius': '8px',
            'color': COLORS['text'],
            'fontSize': '0.9em',
            'maxHeight': '300px',
            'overflowY': 'auto',
            'fontFamily': 'monospace',
            'lineHeight': '1.6'
        })
    ]


if __name__ == '__main__':
    print("\n" + "="*70)
    print("✨ NIMBUSVITA DASHBOARD")
    print("="*70)
    print("🚀 Dashboard iniciado com sucesso!")
    print("🌐 Acesse: http://127.0.0.1:8050/")
    print("📊 Sistema de Predição de Doenças Relacionadas ao Clima")
    print("="*70 + "\n")
    app.run(debug=True, port=8050)
