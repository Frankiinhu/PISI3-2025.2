"""ML Clustering tab layout and callbacks - Clustering models only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc

from ..components import create_card
from ..core.data_context import (
    get_cluster_feature_frame,
    get_context,
)
from ..core.theme import COLORS, page_header, error_figure


# ==================== CACHE VARIABLES ====================
_CACHED_ELBOW_K: int | None = None
_CACHED_CLUSTER_PREP: tuple[int, tuple[int, ...], any] | None = None


# ==================== HELPER FUNCTIONS ====================


def _prepare_cluster_dataset() -> tuple[any, pd.DataFrame, np.ndarray]:
    """Prepare and cache cluster dataset."""
    ctx = get_context()
    try:
        feature_frame = get_cluster_feature_frame()
    except ValueError as exc:
        raise ValueError(f"Erro ao preparar dados: {exc}")
    
    # Cache prepared dataset to avoid repeated scaler.transform / dataframe selection
    global _CACHED_CLUSTER_PREP
    df_key = (feature_frame.shape, tuple(feature_frame.columns))
    ctx_id = id(ctx)
    
    if _CACHED_CLUSTER_PREP is not None:
        cached_id, cached_key, cached_data = _CACHED_CLUSTER_PREP
        if cached_id == ctx_id and cached_key == df_key:
            return cached_data

    scaler = getattr(ctx.clusterer, 'scaler', None)
    if scaler is None:
        raise ValueError("Scaler do clusterizador não encontrado.")

    X_scaled = scaler.transform(feature_frame)
    _CACHED_CLUSTER_PREP = (ctx_id, df_key, (ctx, feature_frame, X_scaled))
    return ctx, feature_frame, X_scaled


def _fit_kmeans_labels(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit KMeans and return labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return model.fit_predict(X_scaled)


def _get_elbow_k(X_scaled: np.ndarray) -> int:
    """Calculate optimal K using elbow method."""
    global _CACHED_ELBOW_K
    if _CACHED_ELBOW_K is not None:
        return _CACHED_ELBOW_K

    max_k = min(10, X_scaled.shape[0])
    scanned_ks: list[int] = []
    inertias: list[float] = []

    for k in range(2, max_k):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X_scaled)
            scanned_ks.append(k)
            inertias.append(model.inertia_)
        except Exception:
            continue

    if not inertias:
        _CACHED_ELBOW_K = 3
        return _CACHED_ELBOW_K

    if len(inertias) < 3:
        _CACHED_ELBOW_K = scanned_ks[0]
        return _CACHED_ELBOW_K

    second_diff = np.diff(inertias, n=2)
    idx = int(np.argmin(second_diff)) + 2
    if idx >= len(scanned_ks):
        _CACHED_ELBOW_K = scanned_ks[-1]
    else:
        _CACHED_ELBOW_K = scanned_ks[idx]
    
    return _CACHED_ELBOW_K


def _prepare_climate_clusters(k: int) -> tuple[pd.DataFrame, list[str]]:
    """Prepare climate clustering data applying KMeans with provided K."""
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
        raise ValueError(f"Colunas ausentes: {', '.join(missing_cols)}")

    climate_vars = ['Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)']
    plot_df = df[required_cols].dropna().copy()
    if plot_df.empty:
        raise ValueError("Sem dados suficientes após limpeza.")

    plot_df = plot_df.rename(columns={diagnosis_col: 'Diagnóstico'})
    mask = plot_df['Diagnóstico'].astype(str).str.upper() != 'H8'
    plot_df = plot_df[mask]
    if plot_df.empty:
        raise ValueError("Sem dados válidos após filtrar diagnósticos.")

    scaler = StandardScaler()
    X_climate = scaler.fit_transform(plot_df[climate_vars])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_climate)
    cluster_series = pd.Series(clusters, index=plot_df.index, dtype=int) + 1
    plot_df['Cluster'] = cluster_series.astype(str)

    return plot_df, climate_vars


# ==================== LAYOUT ====================

def create_layout() -> html.Div:
    """Create ML Clustering tab layout."""
    return dcc.Loading(
        id="loading-ml-clustering",
        type="cube",
        color=COLORS['primary'],
        children=_create_clustering_content()
    )


def _create_clustering_content() -> html.Div:
    """Create the actual clustering content."""
    ctx = get_context()
    
    # Check if clusterer is available
    clusterer_available = (
        ctx.clusterer is not None 
        and getattr(ctx.clusterer, 'model', None) is not None
    )
    
    if not clusterer_available:
        return html.Div([
            page_header(
                '🔬 Modelos ML - Clusterização',
                'Análise de Padrões e Agrupamentos',
                'O modelo de clusterização não foi treinado. Execute o treinamento primeiro.'
            ),
            html.Div([
                dbc.Alert(
                    [
                        html.Span('⚠️', style={'fontSize': '1.3em', 'marginRight': '12px'}),
                        html.Div([
                            html.Strong('Clusterizador não disponível', style={'display': 'block'}),
                            html.Span('Execute o treinamento do modelo de clusterização para usar esta aba.')
                        ], style={'display': 'inline-block'})
                    ],
                    color='warning',
                    style={
                        'backgroundColor': f'rgba(255, 193, 7, 0.15)',
                        'borderLeft': '4px solid #FFC107',
                        'borderRadius': '8px'
                    }
                )
            ], style={'padding': '20px'})
        ])
    
    return html.Div([
        page_header(
            '🔬 Modelos ML - Clusterização',
            'Análise de Padrões e Agrupamentos',
            'Explore clusters de doenças e padrões climáticos'
        ),
        
        # Cluster Summary Card
        html.Div([
            create_card([
                html.Div(id='cluster-summary', children=[
                    html.P('Carregando informações do cluster...', style={'color': COLORS['text_secondary']})
                ])
            ], '📊 Resumo do Modelo de Clusterização')
        ], style={'marginBottom': '20px'}),
        
        # PCA 3D Visualization
        html.Div([
            create_card([
                dcc.Graph(id='cluster-pca-3d-graph')
            ], '🎨 Visualização 3D - PCA dos Clusters')
        ], style={'marginBottom': '20px'}),
        
        # Climate-based Clustering Section
        html.Div([
            html.H3('🌡️ Clusterização Baseada em Variáveis Climáticas', style={
                'color': COLORS['text'],
                'marginBottom': '20px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS["accent"]}',
                'paddingLeft': '14px',
            }),
            
            # K Slider
            html.Div([
                html.Label('Número de Clusters (K):', style={
                    'color': COLORS['text'],
                    'fontWeight': '600',
                    'marginBottom': '10px',
                    'display': 'block'
                }),
                dcc.Slider(
                    id='climate-cluster-k-slider',
                    min=2,
                    max=8,
                    step=1,
                    value=3,
                    marks={i: str(i) for i in range(2, 9)},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ], style={
                'padding': '20px',
                'backgroundColor': COLORS['secondary'],
                'borderRadius': '8px',
                'marginBottom': '20px'
            }),
            
            # Climate 3D Cluster
            create_card([
                dcc.Graph(id='climate-cluster-3d-graph')
            ], '🌍 Clusters Climáticos 3D'),
            
            # Stacked Charts
            html.Div([
                html.Div([
                    create_card([
                        dcc.Graph(id='cluster-diagnosis-stacked')
                    ], '📊 Distribuição de Diagnósticos por Cluster (Modelo Principal)')
                ], style={'flex': '1', 'minWidth': '400px'}),
                
                html.Div([
                    create_card([
                        dcc.Graph(id='climate-cluster-diagnosis-stacked')
                    ], '🌤️ Distribuição de Diagnósticos por Cluster Climático')
                ], style={'flex': '1', 'minWidth': '400px'}),
            ], style={
                'display': 'flex',
                'gap': '20px',
                'flexWrap': 'wrap',
                'marginTop': '20px'
            })
        ], style={'marginTop': '30px'}),
    ], style={'padding': '20px'})


# ==================== CALLBACKS ====================

@callback(
    Output('cluster-summary', 'children'),
    Input('tabs', 'value')
)
def update_cluster_summary(tab):
    """Update cluster model summary."""
    if tab != 'tab-clustering':
        return html.P('Aguardando carregamento...', style={'color': COLORS['text_secondary']})
    
    ctx = get_context()
    clusterer = ctx.clusterer
    
    if clusterer is None or clusterer.model is None:
        return dbc.Alert('Modelo de clusterização não disponível.', color='warning')
    
    n_clusters = len(np.unique(clusterer.model.labels_))
    n_features = len(clusterer.feature_names) if clusterer.feature_names else 'N/A'
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6('Número de Clusters', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em'}),
                    html.H4(str(n_clusters), style={'color': COLORS['primary'], 'fontWeight': '700'})
                ], style={
                    'padding': '16px',
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '8px',
                    'textAlign': 'center'
                })
            ], md=4),
            
            dbc.Col([
                html.Div([
                    html.H6('Features Utilizadas', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em'}),
                    html.H4(str(n_features), style={'color': COLORS['accent'], 'fontWeight': '700'})
                ], style={
                    'padding': '16px',
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '8px',
                    'textAlign': 'center'
                })
            ], md=4),
            
            dbc.Col([
                html.Div([
                    html.H6('Algoritmo', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em'}),
                    html.H4('K-Means', style={'color': COLORS['success'], 'fontWeight': '700'})
                ], style={
                    'padding': '16px',
                    'backgroundColor': COLORS['background'],
                    'borderRadius': '8px',
                    'textAlign': 'center'
                })
            ], md=4),
        ])
    ])


@callback(
    Output('cluster-pca-3d-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_pca_3d(tab):
    """Update PCA 3D cluster visualization."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except ValueError as exc:
        return error_figure(str(exc))
    
    # Perform PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get cluster labels
    labels = ctx.clusterer.model.labels_
    
    # Create 3D scatter
    fig = go.Figure()
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        fig.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(
                size=5,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>Cluster %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>',
            text=[cluster_id] * sum(mask)
        ))
    
    explained_var = pca.explained_variance_ratio_
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
            bgcolor=COLORS['background']
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(30, 33, 57, 0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        title=f'Variância Explicada Total: {sum(explained_var)*100:.1f}%'
    )
    
    return fig


@callback(
    Output('climate-cluster-3d-graph', 'figure'),
    [Input('tabs', 'value'), Input('climate-cluster-k-slider', 'value')]
)
def update_climate_cluster_3d(tab, k):
    """Update climate-based 3D cluster visualization."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        plot_df, climate_vars = _prepare_climate_clusters(k)
    except ValueError as exc:
        return error_figure(str(exc))
    
    fig = go.Figure()
    
    for cluster_id in sorted(plot_df['Cluster'].unique()):
        cluster_data = plot_df[plot_df['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter3d(
            x=cluster_data[climate_vars[0]],
            y=cluster_data[climate_vars[1]],
            z=cluster_data[climate_vars[2]],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(size=4, opacity=0.6),
            hovertemplate='<b>Cluster %{text}</b><br>Temp: %{x:.1f}°C<br>Umidade: %{y:.2f}<br>Vento: %{z:.1f} km/h<extra></extra>',
            text=[cluster_id] * len(cluster_data)
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Temperatura (°C)',
            yaxis_title='Umidade',
            zaxis_title='Velocidade do Vento (km/h)',
            bgcolor=COLORS['background']
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(30, 33, 57, 0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        )
    )
    
    return fig


@callback(
    Output('cluster-diagnosis-stacked', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_diagnosis_stacked(tab):
    """Update diagnosis distribution by main cluster model."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except ValueError as exc:
        return error_figure(str(exc))
    
    labels = ctx.clusterer.model.labels_
    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
    
    # Match cluster labels with diagnosis
    df_plot = ctx.df.loc[feature_frame.index].copy()
    df_plot['Cluster'] = labels
    
    # Count diagnoses per cluster
    cluster_diag = df_plot.groupby(['Cluster', diagnosis_col]).size().reset_index(name='Count')
    
    fig = go.Figure()
    
    for diag in cluster_diag[diagnosis_col].unique():
        diag_data = cluster_diag[cluster_diag[diagnosis_col] == diag]
        fig.add_trace(go.Bar(
            x=diag_data['Cluster'],
            y=diag_data['Count'],
            name=diag,
            hovertemplate='<b>%{fullData.name}</b><br>Cluster: %{x}<br>Casos: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='stack',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        xaxis_title='Cluster',
        yaxis_title='Número de Casos',
        height=500,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(30, 33, 57, 0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        )
    )
    
    return fig


@callback(
    Output('climate-cluster-diagnosis-stacked', 'figure'),
    [Input('tabs', 'value'), Input('climate-cluster-k-slider', 'value')]
)
def update_climate_cluster_diagnosis_stacked(tab, k):
    """Update diagnosis distribution by climate clusters."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        plot_df, _ = _prepare_climate_clusters(k)
    except ValueError as exc:
        return error_figure(str(exc))
    
    cluster_diag = plot_df.groupby(['Cluster', 'Diagnóstico']).size().reset_index(name='Count')
    
    fig = go.Figure()
    
    for diag in cluster_diag['Diagnóstico'].unique():
        diag_data = cluster_diag[cluster_diag['Diagnóstico'] == diag]
        fig.add_trace(go.Bar(
            x=diag_data['Cluster'],
            y=diag_data['Count'],
            name=diag,
            hovertemplate='<b>%{fullData.name}</b><br>Cluster: %{x}<br>Casos: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='stack',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        xaxis_title='Cluster Climático',
        yaxis_title='Número de Casos',
        height=500,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(30, 33, 57, 0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        )
    )
    
    return fig


def register_callbacks(app):
    """Register clustering callbacks with the app.
    
    Note: Callbacks are registered using @callback decorator,
    so this function is just a placeholder for consistency.
    """
    pass


__all__ = ['create_layout', 'register_callbacks']
