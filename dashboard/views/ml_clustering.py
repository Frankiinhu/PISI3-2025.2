"""ML Clustering tab layout and callbacks - Clustering models only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Tuple, Any, Literal
from dash import Input, Output, dcc, html, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import dash_bootstrap_components as dbc

from ..components import create_card
from ..core.data_context import (
    get_cluster_feature_frame,
    get_context,
    get_context_version,
    get_model_status,
)
from ..core.theme import COLORS, page_header, error_figure
from ..utils.ui import alert_component
from src.models.clustering import DiseaseClusterer


# ==================== HELPER FUNCTIONS ====================


@lru_cache(maxsize=4)
def _cached_cluster_prep(version_key: tuple | None) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """Cache prepared clustering data based on dataset/model mtimes."""
    ctx = get_context()
    if ctx.clusterer is None:
        raise ValueError("Clusterizador não disponível. Execute: python scripts/train_models.py --use-kmodes")

    if ctx.clusterer.feature_names is None:
        raise ValueError("Modelo de clusterização não possui feature_names. Retreine o modelo.")

    feature_frame = get_cluster_feature_frame()
    clusterer = ctx.clusterer
    X_scaled = clusterer.prepare_data(feature_frame)
    return feature_frame, X_scaled, clusterer.mode


def _prepare_cluster_dataset() -> Tuple[Any, pd.DataFrame, np.ndarray]:
    """Prepare and cache cluster dataset using original feature data (not PCA)."""
    try:
        ctx = get_context()
        version_key = get_context_version()
        feature_frame, X_scaled, _ = _cached_cluster_prep(version_key)
        return ctx, feature_frame, X_scaled
    except Exception as exc:
        raise ValueError(f"Erro ao preparar dados para clusterização: {exc}") from exc


@lru_cache(maxsize=4)
def _get_optimal_k_analysis(version_key: tuple | None, mode: Literal['kmeans', 'kmodes'] = 'kmodes') -> dict:
    """Get optimal K analysis using DiseaseClusterer methods with caching."""
    _, X_scaled, cached_mode = _cached_cluster_prep(version_key)
    resolved_mode = mode or cached_mode

    temp_clusterer = DiseaseClusterer(random_state=42, mode=resolved_mode)
    return temp_clusterer.find_optimal_clusters(X_scaled, max_clusters=10, sample_size=1000)


@lru_cache(maxsize=4)
def _cached_pca_3d(version_key: tuple | None) -> Tuple[np.ndarray, np.ndarray]:
    """Cache PCA 3D reduction for clustering visualization."""
    _, X_scaled, _ = _cached_cluster_prep(version_key)
    pca = PCA(n_components=3, random_state=42)
    return pca.fit_transform(X_scaled), pca.explained_variance_ratio_


def _get_hover_info(df: pd.DataFrame, indices: np.ndarray, ctx) -> list[str]:
    """Cria informações de hover detalhadas para cada ponto.
    
    Args:
        df: DataFrame original
        indices: Índices dos pontos
        ctx: Contexto com informações do dataset
        
    Returns:
        Lista de strings formatadas para hover
    """
    hover_texts = []
    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
    
    for idx in indices:
        if idx >= len(df):
            hover_texts.append("Índice inválido")
            continue
            
        row = df.iloc[idx]
        
        # Diagnóstico
        diagnosis = row.get(diagnosis_col, 'N/A')
        
        # Demografia
        age = row.get('Idade', 'N/A')
        gender = row.get('Gênero', row.get('Sexo', 'N/A'))
        
        # Sintomas principais (colunas binárias)
        symptoms = []
        for col in df.columns:
            if col.startswith(('Febre', 'Tosse', 'Dor', 'Falta', 'Náusea', 'Diarreia', 
                              'Fadiga', 'Coriza', 'Espirro', 'Congestão')):
                if row.get(col, 0) == 1:
                    symptoms.append(col.replace('_', ' '))
        
        symptoms_str = ', '.join(symptoms[:3]) if symptoms else 'Nenhum registrado'
        if len(symptoms) > 3:
            symptoms_str += f' (+{len(symptoms)-3})'
        
        # Fatores climáticos
        temp = row.get('Temperatura (°C)', 'N/A')
        humidity = row.get('Umidade', 'N/A')
        
        hover_text = (
            f"<b>Diagnóstico: {diagnosis}</b><br>"
            f"Idade: {age} | Gênero: {gender}<br>"
            f"Sintomas: {symptoms_str}<br>"
            f"Temp: {temp}°C | Umidade: {humidity}"
        )
        hover_texts.append(hover_text)
    
    return hover_texts


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

def create_layout():
    """Create ML Clustering tab layout."""
    return dcc.Loading(
        id="loading-ml-clustering",
        type="cube",
        color=COLORS['primary'],
        children=_create_clustering_content()
    )


def _create_clustering_content() -> html.Div:
    """Create the actual clustering content."""
    try:
        ctx = get_context()
    except Exception as exc:
        return html.Div([
            page_header(
                '🔬 Modelos ML - Clusterização',
                'Análise de Padrões e Agrupamentos',
                'Erro ao carregar contexto de dados'
            ),
            html.Div([
                dbc.Alert(
                    f"Erro: {str(exc)}",
                    color='danger',
                    style={'margin': '20px'}
                )
            ])
        ])
    
    # Check if clusterer is available
    clusterer_available = (
        ctx.clusterer is not None 
        and getattr(ctx.clusterer, 'model', None) is not None
        and getattr(ctx.clusterer, 'feature_names', None) is not None
    )
    
    if not clusterer_available:
        model_status = get_model_status()
        error_detail = model_status.get('clusterer')
        message = 'Execute o treinamento do modelo de clusterização para usar esta aba.'
        if error_detail:
            message = f'{message} Detalhe: {error_detail}'

        return html.Div([
            page_header(
                '🔬 Modelos ML - Clusterização',
                'Análise de Padrões e Agrupamentos',
                'O modelo de clusterização não foi treinado. Execute o treinamento primeiro.'
            ),
            html.Div([
                alert_component('warning', 'Clusterizador não disponível', message)
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
        
        # K Selection Methods Visualization
        html.Div([
            html.H3('📊 Seleção do Número Ótimo de Clusters (K)', style={
                'color': COLORS['text'],
                'marginBottom': '20px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS["primary"]}',
                'paddingLeft': '14px',
            }),
            
            html.Div([
                html.Div([
                    create_card([
                        dcc.Graph(id='cluster-elbow-graph')
                    ], '📐 Método do Cotovelo (Elbow)')
                ], style={'flex': '1', 'minWidth': '400px'}),
                
                html.Div([
                    create_card([
                        dcc.Graph(id='cluster-silhouette-graph')
                    ], '📏 Análise de Silhueta')
                ], style={'flex': '1', 'minWidth': '400px'}),
            ], style={
                'display': 'flex',
                'gap': '20px',
                'flexWrap': 'wrap',
                'marginBottom': '20px'
            })
        ], style={'marginBottom': '30px'}),
        
        # PCA 3D Visualizations - Both K methods
        html.Div([
            html.H3('🎨 Visualizações 3D - PCA dos Clusters', style={
                'color': COLORS['text'],
                'marginBottom': '20px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS["accent"]}',
                'paddingLeft': '14px',
            }),
            
            html.Div([
                html.Div([
                    create_card([
                        dcc.Graph(id='cluster-pca-3d-elbow-graph')
                    ], '📐 PCA 3D - K Cotovelo')
                ], style={'flex': '1', 'minWidth': '500px'}),
                
                html.Div([
                    create_card([
                        dcc.Graph(id='cluster-pca-3d-silhouette-graph')
                    ], '📏 PCA 3D - K Silhueta')
                ], style={'flex': '1', 'minWidth': '500px'}),
            ], style={
                'display': 'flex',
                'gap': '20px',
                'flexWrap': 'wrap',
                'marginBottom': '20px'
            })
        ], style={'marginBottom': '30px'}),
        
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
    
    try:
        ctx = get_context()
    except Exception as exc:
        return dbc.Alert(f'Erro ao carregar dados: {str(exc)}', color='danger')
    
    clusterer = ctx.clusterer
    
    if clusterer is None or clusterer.model is None:
        return dbc.Alert('Modelo de clusterização não disponível.', color='warning')
    
    # Get labels safely
    if hasattr(clusterer.model, 'labels_') and clusterer.model.labels_ is not None:
        n_clusters = len(np.unique(clusterer.model.labels_))
    elif hasattr(clusterer.model, 'n_clusters'):
        n_clusters = clusterer.model.n_clusters
    else:
        n_clusters = 'N/A'
    
    n_features = len(clusterer.feature_names) if clusterer.feature_names else 'N/A'
    algorithm_name = 'K-Modes' if clusterer.mode == 'kmodes' else 'K-Means'
    
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
                    html.H4(algorithm_name, style={'color': COLORS['success'], 'fontWeight': '700'})
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
    Output('cluster-elbow-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_elbow(tab):
    """Update elbow method graph using DiseaseClusterer."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except ValueError as exc:
        return error_figure(str(exc))
    
    # Get optimal K analysis from clusterer
    version_key = get_context_version()
    results = _get_optimal_k_analysis(version_key)
    
    ks = results['ks']
    inertias = results['inertias']
    elbow_k = results.get('elbow_k')
    
    if elbow_k is None:
        elbow_k = ks[0] if ks else 3
    
    fig = go.Figure()
    
    # Line plot
    fig.add_trace(go.Scatter(
        x=ks,
        y=inertias,
        mode='lines+markers',
        name='Inércia',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, color=COLORS['primary'], line=dict(width=2, color='white')),
        hovertemplate='<b>K=%{x}</b><br>Inércia: %{y:.2f}<extra></extra>'
    ))
    
    # Highlight elbow point
    if elbow_k in ks:
        elbow_idx = ks.index(elbow_k)
        elbow_inertia = inertias[elbow_idx]
        fig.add_trace(go.Scatter(
            x=[elbow_k],
            y=[elbow_inertia],
            mode='markers',
            name=f'Cotovelo (K={elbow_k})',
            marker=dict(size=15, color=COLORS['accent'], symbol='star', line=dict(width=2, color='white')),
            hovertemplate=f'<b>K Ótimo: {elbow_k}</b><br>Inércia: {elbow_inertia:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        xaxis_title='Número de Clusters (K)',
        yaxis_title='Inércia (Within-Cluster Sum of Squares)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        height=400,
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(30, 33, 57, 0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        xaxis=dict(gridcolor=COLORS['border'], dtick=1),
        yaxis=dict(gridcolor=COLORS['border']),
        hovermode='x unified'
    )
    
    return fig


@callback(
    Output('cluster-silhouette-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_silhouette(tab):
    """Update silhouette analysis graph using DiseaseClusterer."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except ValueError as exc:
        return error_figure(str(exc))
    
    # Get optimal K analysis from clusterer
    version_key = get_context_version()
    results = _get_optimal_k_analysis(version_key)
    
    ks = results['ks']
    silhouette_scores = results['silhouette_scores']
    best_k = results.get('silhouette_best_k')
    
    # Filter out NaN scores
    valid_data = [(k, s) for k, s in zip(ks, silhouette_scores) if not np.isnan(s)]
    
    if not valid_data:
        return error_figure("Não foi possível calcular scores de silhueta válidos.")
    
    valid_ks, valid_scores = zip(*valid_data)
    
    if best_k is None and valid_scores:
        best_k = valid_ks[int(np.argmax(valid_scores))]
    
    fig = go.Figure()
    
    # Bar plot
    colors = [COLORS['accent'] if k == best_k else COLORS['primary'] for k in valid_ks]
    
    fig.add_trace(go.Bar(
        x=valid_ks,
        y=valid_scores,
        marker=dict(color=colors, line=dict(width=2, color='white')),
        hovertemplate='<b>K=%{x}</b><br>Silhueta: %{y:.4f}<extra></extra>',
        name='Score de Silhueta'
    ))
    
    # Add annotation for best K
    if best_k in valid_ks:
        best_idx = list(valid_ks).index(best_k)
        best_score = valid_scores[best_idx]
        fig.add_annotation(
            x=best_k,
            y=best_score,
            text=f'Melhor K={best_k}',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=COLORS['accent'],
            ax=0,
            ay=-40,
            font=dict(size=12, color=COLORS['text'], weight='bold'),
            bgcolor='rgba(30, 33, 57, 0.9)',
            bordercolor=COLORS['accent'],
            borderwidth=2,
            borderpad=4
        )
    
    fig.update_layout(
        xaxis_title='Número de Clusters (K)',
        yaxis_title='Score de Silhueta (maior é melhor)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        height=400,
        showlegend=False,
        xaxis=dict(gridcolor=COLORS['border'], dtick=1),
        yaxis=dict(gridcolor=COLORS['border'], range=[0, 1]),
        hovermode='x unified'
    )
    
    return fig


@callback(
    Output('cluster-pca-3d-elbow-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_pca_3d_elbow(tab):
    """Update PCA 3D cluster visualization using elbow K."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except ValueError as exc:
        return error_figure(str(exc))
    
    # Get optimal K analysis from clusterer
    version_key = get_context_version()
    results = _get_optimal_k_analysis(version_key, mode='kmodes')
    elbow_k = results.get('elbow_k')
    
    if elbow_k is None:
        elbow_k = 3  # Default fallback
    
    # Fit KMeans with elbow K
    model = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    # Perform PCA (cached)
    X_pca, explained_var = _cached_pca_3d(version_key)
    
    # Get hover information
    hover_texts = _get_hover_info(ctx.df, feature_frame.index.to_numpy(), ctx)
    
    # Create 3D scatter
    fig = go.Figure()
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
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
            hovertemplate='%{text}<extra></extra>',
            text=[hover_texts[i] for i in cluster_indices]
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
            bgcolor=COLORS['secondary'],
            xaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border']),
            yaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border']),
            zaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border'])
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
        title=dict(
            text=f'K={elbow_k} (Método Cotovelo - K-Modes) | Variância Explicada: {sum(explained_var)*100:.1f}%',
            font=dict(size=14, color=COLORS['text'])
        )
    )
    
    return fig


@callback(
    Output('cluster-pca-3d-silhouette-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_pca_3d_silhouette(tab):
    """Update PCA 3D cluster visualization using silhouette K."""
    if tab != 'tab-clustering':
        return go.Figure()
    
    try:
        ctx, feature_frame, X_scaled = _prepare_cluster_dataset()
    except ValueError as exc:
        return error_figure(str(exc))
    
    # Get optimal K analysis from clusterer
    version_key = get_context_version()
    results = _get_optimal_k_analysis(version_key, mode='kmodes')
    silhouette_k = results.get('silhouette_best_k')
    
    if silhouette_k is None:
        silhouette_k = 3  # Default fallback
    
    # Fit KMeans with silhouette K
    model = KMeans(n_clusters=silhouette_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    # Perform PCA (cached)
    X_pca, explained_var = _cached_pca_3d(version_key)
    
    # Get hover information
    hover_texts = _get_hover_info(ctx.df, feature_frame.index.to_numpy(), ctx)
    
    # Create 3D scatter
    fig = go.Figure()
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
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
            hovertemplate='%{text}<extra></extra>',
            text=[hover_texts[i] for i in cluster_indices]
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
            bgcolor=COLORS['secondary'],
            xaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border']),
            yaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border']),
            zaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border'])
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
        title=dict(
            text=f'K={silhouette_k} (Método Silhueta - K-Modes) | Variância Explicada: {sum(explained_var)*100:.1f}%',
            font=dict(size=14, color=COLORS['text'])
        )
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
    
    # Get DataContext for hover information
    ctx = get_context()
    
    fig = go.Figure()
    
    for cluster_id in sorted(plot_df['Cluster'].unique()):
        cluster_data = plot_df[plot_df['Cluster'] == cluster_id]
        cluster_indices = cluster_data.index.tolist()
        
        # Get hover information for each point
        hover_texts = _get_hover_info(ctx.df, cluster_indices, ctx)
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data[climate_vars[0]],
            y=cluster_data[climate_vars[1]],
            z=cluster_data[climate_vars[2]],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(size=4, opacity=0.6),
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Temperatura (°C)',
            yaxis_title='Umidade',
            zaxis_title='Velocidade do Vento (km/h)',
            bgcolor=COLORS['secondary'],
            xaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border']),
            yaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border']),
            zaxis=dict(backgroundcolor=COLORS['secondary'], gridcolor=COLORS['border'])
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
    
    # Get labels from the model or predict
    if hasattr(ctx.clusterer.model, 'labels_') and ctx.clusterer.model.labels_ is not None:
        labels = ctx.clusterer.model.labels_
    else:
        # Predict labels if not available
        labels = ctx.clusterer.predict_cluster(X_scaled)
    
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
