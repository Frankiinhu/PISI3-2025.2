"""Merged Pipeline and Hyperparameter Tuning tab layout and callbacks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from ..components import create_card
from ..core.data_context import get_context, is_classifier_available
from ..core.theme import COLORS, page_header, error_figure as _empty_figure
from ..utils.ui import section_header



def create_layout() -> html.Div:
    """Create merged pipeline and tuning tab layout."""
    return dcc.Loading(
        id="loading-pipeline-tuning",
        type="cube",
        color=COLORS['primary'],
        children=_create_pipeline_content()
    )


def _create_pipeline_content() -> html.Div:
    """Create the actual pipeline content."""
    return html.Div([
        page_header(
            '🔧 Pipeline de Treinamento & Tuning',
            'Fluxo completo de treinamento com SMOTE e otimização de hiperparâmetros',
            'Visualize o pipeline end-to-end, compare 4 modelos com/sem SMOTE e veja parâmetros otimizados'
        ),
        
        # Pipeline Flow
        section_header(
            '📊 Fluxo do Pipeline',
            'Etapas do processo de treinamento de modelos'
        ),
        
        html.Div([
            create_card([
                dcc.Graph(id='pipeline-flow-graph', config={'displayModeBar': False})
            ], '🔄 Fluxo de Processamento')
        ], style={'marginBottom': '20px'}),
        
        # Controls and Status
        html.Div([
            html.Div([
                create_card([
                    html.Div([
                        html.Label('🎛️ Opções de Treinamento', style={
                            'fontWeight': '600',
                            'marginBottom': '15px',
                            'color': COLORS['text'],
                            'fontSize': '1.1em'
                        }),
                        dcc.Checklist(
                            id='smote-checklist',
                            options=[{'label': ' Usar SMOTE (balanceamento de classes)', 'value': 'use_smote'}],
                            value=[],
                            style={'color': COLORS['text'], 'fontSize': '1em'},
                            inputStyle={'marginRight': '10px', 'cursor': 'pointer'}
                        ),
                        html.P(
                            'SMOTE compara 4 modelos (LR, SVC, RF, GB) com e sem balanceamento.',
                            style={
                                'color': COLORS['text_secondary'],
                                'fontSize': '0.85em',
                                'marginTop': '10px',
                                'fontStyle': 'italic'
                            }
                        ),
                    ])
                ], '⚙️ Controles'),
            ], style={'flex': '1', 'minWidth': '300px'}),
            
            html.Div([
                create_card([
                    html.Div(id='pipeline-status-display')
                ], '📊 Status da Execução'),
            ], style={'flex': '2', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px', 'flexWrap': 'wrap'}),
        
        # Tuning Status
        html.Div(id='tuning-status-section', children=[]),
        
        # Best Params
        html.Div(id='best-params-section', children=[]),
        
        # Model Comparison
        section_header(
            '🔍 Comparação de Modelos (4 Algoritmos)',
            'Performance com e sem SMOTE quando habilitado'
        ),
        
        html.Div([
            create_card([
                dcc.Graph(id='pipeline-model-comparison', config={'displayModeBar': False})
            ], '📈 Acurácia Balanceada dos 4 Modelos'),
            
            html.Div(
                create_card([
                    dcc.Graph(id='pipeline-model-comparison-table', config={'displayModeBar': False})
                ], '📋 Tabela Detalhada de Métricas'),
                style={'marginTop': '20px'}
            )
        ], style={'marginBottom': '20px'}),
    ])


def register_callbacks(app) -> None:
    """Register all callbacks for pipeline and tuning tab."""
    
    @app.callback(
        [Output('pipeline-flow-graph', 'figure'),
         Output('pipeline-model-comparison', 'figure'),
         Output('pipeline-model-comparison-table', 'figure'),
         Output('tuning-status-section', 'children'),
         Output('best-params-section', 'children'),
         Output('pipeline-status-display', 'children')],
        [Input('tabs', 'value'),
         Input('smote-checklist', 'value')]
    )
    def update_pipeline_visualizations(tab, smote_checked):
        """Update pipeline visualizations with real data from classifier."""
        if tab != 'tab-pipeline-tuning':
            return go.Figure(), go.Figure(), go.Figure(), html.Div(), html.Div(), html.Div()
        
        ctx = get_context()
        classifier_available = is_classifier_available()
        
        # 1. Pipeline Flow
        flow_fig = _create_flow_diagram()
        
        # 2. Model Comparison
        comparison_fig = _create_model_comparison_graph(ctx, classifier_available, smote_checked)
        
        # 3. Model Comparison Table
        comparison_table_fig = _create_model_comparison_table(ctx, classifier_available, smote_checked)
        
        # 4. Tuning Status
        tuning_status_div = _create_tuning_status_section(ctx, classifier_available)
        
        # 5. Best Parameters
        best_params_div = _create_best_params_section(ctx, classifier_available)
        
        # 6. Status display
        status_div = _create_status_display(ctx, classifier_available, smote_checked)
        
        return flow_fig, comparison_fig, comparison_table_fig, tuning_status_div, best_params_div, status_div


def _create_flow_diagram() -> go.Figure:
    """Create pipeline flow network diagram."""
    fig = go.Figure()
    
    stages_info = [
        {"name": "📥 Dados<br>Brutos", "x": 0.1, "y": 0.5, "color": "#a4a8ff", "size": 50},
        {"name": "🧹 Limpeza<br>de Dados", "x": 0.25, "y": 0.8, "color": "#5559ff", "size": 45},
        {"name": "🔧 Feature<br>Engineering", "x": 0.4, "y": 0.8, "color": "#7b7fff", "size": 45},
        {"name": "📊 Train/Test<br>Split", "x": 0.55, "y": 0.5, "color": "#4facfe", "size": 48},
        {"name": "🤖 Treinamento<br>de Modelos", "x": 0.7, "y": 0.2, "color": "#a4a8ff", "size": 55},
        {"name": "✅ Validação<br>Cruzada", "x": 0.7, "y": 0.8, "color": "#4ade80", "size": 45},
        {"name": "💾 Modelo<br>Salvo", "x": 0.9, "y": 0.5, "color": "#fbbf24", "size": 50},
    ]
    
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
    
    # Draw connections
    for source, target in edges:
        fig.add_trace(go.Scatter(
            x=[stages_info[source]["x"], stages_info[target]["x"]],
            y=[stages_info[source]["y"], stages_info[target]["y"]],
            mode='lines',
            line=dict(color='rgba(164, 168, 255, 0.3)', width=3),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Draw nodes
    for stage in stages_info:
        fig.add_trace(go.Scatter(
            x=[stage["x"]],
            y=[stage["y"]],
            mode='markers+text',
            marker=dict(size=stage["size"], color=stage["color"], line=dict(color='white', width=2)),
            text=stage["name"],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Inter, sans-serif', weight='bold'),
            hovertemplate=f'<b>{stage["name"]}</b><extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color=COLORS['text'], family='Inter, sans-serif'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        margin=dict(t=30, b=30, l=30, r=30),
        hovermode='closest'
    )
    
    return fig


def _create_model_comparison_graph(ctx, classifier_available: bool, smote_checked) -> go.Figure:
    """Create 4-model comparison graph (LR, SVC, RF, GB) with/without SMOTE."""
    use_smote = 'use_smote' in (smote_checked or [])
    
    if not classifier_available:
        return _empty_figure('Classificador não disponível.<br>Treine o modelo primeiro.')
    
    clf = ctx.classifier
    
    # Check if we have SMOTE comparison results
    if use_smote and hasattr(clf, 'smote_comparison_results') and clf.smote_comparison_results:
        results = clf.smote_comparison_results
        df_base = results['base']
        df_smote = results['smote']
        
        # Models to compare
        models = df_base['Modelo'].tolist()
        base_acc = df_base['Acurácia Balanceada'].tolist()
        smote_acc = df_smote['Acurácia Balanceada'].tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Base (Sem SMOTE)',
            x=models,
            y=base_acc,
            marker_color=COLORS['primary'],
            text=[f'{v:.1%}' for v in base_acc],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Base: %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Com SMOTE',
            x=models,
            y=smote_acc,
            marker_color=COLORS['accent'],
            text=[f'{v:.1%}' for v in smote_acc],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>SMOTE: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            barmode='group',
            title='Comparação: 4 Modelos (LR, SVC, RF, GB) - Base vs SMOTE',
            xaxis_title='Modelo',
            yaxis_title='Acurácia Balanceada',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text'], family='Inter, sans-serif'),
            height=500,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis=dict(tickformat='.0%', range=[0, 1.1])
        )
        
        return fig
    
    # Fallback: show current model only
    metrics = clf.metrics
    if metrics:
        model_name = type(clf.model).__name__ if clf.model else 'Modelo Atual'
        
        fig = go.Figure(go.Bar(
            x=[model_name],
            y=[metrics.get('balanced_accuracy', 0)],
            marker_color=COLORS['primary'],
            text=[f"{metrics.get('balanced_accuracy', 0):.1%}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Métricas do {model_name}<br><sub>Habilite SMOTE para comparação de 4 modelos</sub>',
            xaxis_title='Modelo',
            yaxis_title='Acurácia Balanceada',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text'], family='Inter, sans-serif'),
            height=500,
            yaxis=dict(tickformat='.0%')
        )
        
        return fig
    
    return _empty_figure('Métricas não disponíveis.<br>Execute treinamento com comparação SMOTE.')


def _create_model_comparison_table(ctx, classifier_available: bool, smote_checked) -> go.Figure:
    """Create detailed table with metrics for all 4 models."""
    use_smote = 'use_smote' in (smote_checked or [])
    
    if not classifier_available:
        return go.Figure()
    
    clf = ctx.classifier
    
    # Try to get SMOTE comparison
    if use_smote and hasattr(clf, 'smote_comparison_results') and clf.smote_comparison_results:
        results = clf.smote_comparison_results
        df_base = results['base'].copy()
        df_smote = results['smote'].copy()
        
        # Combine
        df_combined = pd.concat([df_base, df_smote], ignore_index=True)
        
        # Format percentages
        for col in ['Acurácia', 'Acurácia Balanceada', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']:
            if col in df_combined.columns:
                df_combined[col] = df_combined[col].apply(lambda x: f'{x:.2%}')
        
        df_display = df_combined[['Modelo', 'Versão', 'Acurácia', 'Acurácia Balanceada', 'F1-Score (Macro)']]
    
    elif hasattr(clf, 'metrics') and clf.metrics:
        # Single model
        model_name = type(clf.model).__name__ if clf.model else 'Modelo Atual'
        metrics = clf.metrics
        df_display = pd.DataFrame([{
            'Modelo': model_name,
            'Acurácia': f"{metrics.get('accuracy', 0):.2%}",
            'Acurácia Balanceada': f"{metrics.get('balanced_accuracy', 0):.2%}",
            'Precision (Macro)': f"{metrics.get('precision_macro', 0):.2%}",
            'F1-Score (Macro)': f"{metrics.get('f1_macro', 0):.2%}"
        }])
    else:
        df_display = pd.DataFrame([{'Modelo': 'N/A', 'Status': 'Sem dados'}])
    
    if df_display.empty:
        return go.Figure()
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>' + col + '</b>' for col in df_display.columns],
            fill_color=COLORS['primary'],
            align='center',
            font=dict(color='white', size=12, family='Inter, sans-serif')
        ),
        cells=dict(
            values=[df_display[col] for col in df_display.columns],
            fill_color=COLORS['card'],
            align='center',
            font=dict(color=COLORS['text'], size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        title='Detalhamento Completo de Métricas',
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        height=max(200, len(df_display) * 40 + 100),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig


def _create_tuning_status_section(ctx, classifier_available: bool) -> html.Div:
    """Create tuning status section if available."""
    if not classifier_available:
        return html.Div()
    
    clf = ctx.classifier
    if not hasattr(clf, 'best_params') or clf.best_params is None:
        return html.Div()
    
    # Extract tuning info
    best_params = clf.best_params
    search_method = best_params.get('_search_method', 'Desconhecido')
    num_params = len([k for k in best_params.keys() if not k.startswith('_')])
    
    return html.Div([
        _section_header('🎯 Status de Tuning', 'Otimização de hiperparâmetros realizada'),
        create_card([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('Método', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em'}),
                        html.H4(search_method, style={'color': COLORS['primary'], 'fontWeight': '700'})
                    ], style={'padding': '16px', 'backgroundColor': COLORS['background'], 'borderRadius': '8px', 'textAlign': 'center'})
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H6('Parâmetros Otimizados', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em'}),
                        html.H4(str(num_params), style={'color': COLORS['accent'], 'fontWeight': '700'})
                    ], style={'padding': '16px', 'backgroundColor': COLORS['background'], 'borderRadius': '8px', 'textAlign': 'center'})
                ], md=6),
            ])
        ], '📊 Informações de Tuning')
    ], style={'marginTop': '30px', 'marginBottom': '20px'})


def _create_best_params_section(ctx, classifier_available: bool) -> html.Div:
    """Display best parameters if tuning was performed."""
    if not classifier_available:
        return html.Div()
    
    clf = ctx.classifier
    if not hasattr(clf, 'best_params') or clf.best_params is None:
        return html.Div()
    
    best_params = clf.best_params
    params_list = [
        html.Li(f"{k}: {v}", style={'marginBottom': '8px'})
        for k, v in best_params.items()
        if not k.startswith('_')
    ]
    
    return html.Div([
        create_card([
            html.H5('Melhores Parâmetros Encontrados', style={'color': COLORS['text'], 'marginBottom': '16px'}),
            html.Ul(params_list, style={'color': COLORS['text_secondary'], 'fontSize': '0.95em'})
        ], '⚙️ Hiperparâmetros Otimizados')
    ], style={'marginBottom': '20px'})


def _create_status_display(ctx, classifier_available: bool, smote_checked) -> html.Div:
    """Display current pipeline status."""
    use_smote = 'use_smote' in (smote_checked or [])
    
    if not classifier_available:
        return html.Div([
            html.P('⚠️ Modelo não treinado', style={'color': COLORS['warning'], 'fontWeight': '600'}),
            html.P('Execute o treinamento para visualizar métricas.', style={'color': COLORS['text_secondary'], 'fontSize': '0.9em'})
        ])
    
    clf = ctx.classifier
    model_name = type(clf.model).__name__ if clf.model else 'Desconhecido'
    
    if use_smote and hasattr(clf, 'smote_comparison_results') and clf.smote_comparison_results:
        return html.Div([
            html.P('✅ Comparação SMOTE disponível', style={'color': COLORS['success'], 'fontWeight': '600'}),
            html.P('4 modelos comparados: LR, SVC, RF, GB', style={'color': COLORS['text_secondary'], 'fontSize': '0.9em'})
        ])
    
    return html.Div([
        html.P(f'✅ Modelo treinado: {model_name}', style={'color': COLORS['success'], 'fontWeight': '600'}),
        html.P('Habilite SMOTE para comparar 4 modelos.', style={'color': COLORS['text_secondary'], 'fontSize': '0.9em'})
    ])


__all__ = ['create_layout', 'register_callbacks']
