"""Classification and SHAP Explainability tab layout and callbacks."""
from __future__ import annotations


import numpy as np
import pandas as pd

from dash import Input, Output, dcc, html, callback, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

from ..components import create_card
from ..core.data_context import (
    get_context,
    get_context_version,
    get_model_status,
    has_feature_importances,
    is_classifier_available,
)
from ..core.theme import COLORS, page_header
from ..utils.ui import graph_card, graph_row, section_header, alert_component
from ..shap_utils import (
    create_shap_feature_importance_bar,
    create_shap_multiclass_bar,
    create_shap_beeswarm,
    create_shap_force_plot,
    get_shap_summary_stats
)


_SHAP_FIGURE_CACHE: dict[tuple, go.Figure] = {}


def _shap_cache_key(ctx, suffix: str, sample_idx: int | None = None) -> tuple:
    clf = ctx.classifier
    feature_names = tuple(clf.feature_names or [])
    return (
        get_context_version(),
        id(clf.shap_values),
        id(clf.shap_data),
        feature_names,
        suffix,
        sample_idx,
    )



def create_layout() -> html.Div:
    """Create classification and SHAP analysis tab layout."""
    return dcc.Loading(
        id="loading-classification",
        type="cube",
        color=COLORS['primary'],
        children=_create_classification_content()
    )


def _create_classification_content() -> html.Div:
    """Create the actual classification content."""
    ctx = get_context()
    classifier_available = is_classifier_available()
    
    if not classifier_available:
        model_status = get_model_status()
        error_detail = model_status.get('classifier')
        message = 'Execute o treinamento do modelo para usar esta aba.'
        if error_detail:
            message = f'{message} Detalhe: {error_detail}'

        return html.Div([
            page_header(
                '🤖 Classificação e Explicabilidade SHAP',
                'Predições, Feature Importance e Análise Local com SHAP',
                'O classificador não foi treinado. Execute o treinamento primeiro.'
            ),
            html.Div([
                alert_component('warning', 'Classificador não disponível', message)
            ], style={'padding': '20px'})
        ])
    
    # Obter informações do classificador
    clf = ctx.classifier
    classes = clf.label_encoder.classes_
    shap_available = clf.shap_values is not None
    
    return html.Div([
        page_header(
            '🤖 Classificação e Explicabilidade SHAP',
            'Predições, Feature Importance e Análise Local com SHAP',
            'Explore explicações globais e locais das predições do modelo'
        ),
        
        # ===== EXPLICAÇÕES GLOBAIS =====
        section_header(
            '📊 Explicações Globais (Model Behavior)',
            'Entenda quais features são mais importantes para o modelo em geral'
        ),
        
        html.Div([
            dbc.Alert(
                [
                    html.Span('ℹ️', style={'fontSize': '1.3em', 'marginRight': '12px'}),
                    html.Div([
                        html.Strong('SHAP - SHapley Additive exPlanations', style={'display': 'block'}),
                        html.Span('Usa a teoria dos jogos para explicar predições de forma consistente e confiável.')
                    ], style={'display': 'inline-block'})
                ],
                color='info',
                style={
                    'backgroundColor': f'rgba(33, 150, 243, 0.15)',
                    'borderLeft': '4px solid #2196F3',
                    'borderRadius': '8px'
                }
            ),
        ], style={'marginBottom': '20px', 'paddingX': '20px'}),
        
        # Feature Importance Bar
        graph_row([
            graph_card('shap-feature-importance', 'Feature Importance (SHAP)', '1 1 100%')
        ]),
        
        # Multiclass Feature Importance (se aplicável)
        html.Div([
            graph_row([
                graph_card('shap-multiclass-bar', 'Importância por Classe (SHAP)', '1 1 100%')
            ]) if len(classes) > 1 else None
        ]),
        
        # Beeswarm Plot
        html.Div([
            graph_row([
                graph_card('shap-beeswarm', 'SHAP Beeswarm Plot', '1 1 100%')
            ])
        ]) if shap_available else None,
        
        # ===== EXPLICAÇÃO LOCAL =====
        section_header(
            '🔍 Explicação Local (Predição Individual)',
            'Selecione uma amostra para entender a predição específica',
            'primary'
        ),
        
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label(
                        '📍 Selecionar Amostra',
                        style={
                            'color': COLORS['text'],
                            'fontWeight': '600',
                            'marginBottom': '8px'
                        }
                    ),
                    dcc.Slider(
                        id='sample-index-slider',
                        min=0,
                        max=99,
                        step=1,
                        value=0,
                        marks={0: '0', 25: '25', 50: '50', 75: '75', 99: '99'},
                        tooltip={'placement': 'bottom', 'always_visible': False}
                    ),
                    html.P(
                        id='sample-index-display',
                        style={
                            'color': COLORS['text_secondary'],
                            'fontSize': '0.9em',
                            'marginTop': '8px'
                        }
                    )
                ], md=12)
            ])
        ], style={'padding': '20px', 'backgroundColor': COLORS['secondary'], 'borderRadius': '8px', 'marginBottom': '20px'}),
        
        # Force Plot (Explicação Local)
        graph_row([
            graph_card('shap-force-plot', 'Force Plot (Waterfall)', '1 1 100%')
        ]) if shap_available else html.Div([
            dbc.Alert(
                [
                    html.Span('⚠️', style={'fontSize': '1.3em', 'marginRight': '12px'}),
                    html.Div([
                        html.Strong('SHAP Values não calculados', style={'display': 'block'}),
                        html.Span('Execute o treinamento com cálculo de SHAP para usar esta feature.')
                    ], style={'display': 'inline-block'})
                ],
                color='warning',
                style={
                    'backgroundColor': f'rgba(255, 193, 7, 0.15)',
                    'borderLeft': '4px solid #FFC107',
                    'borderRadius': '8px'
                }
            )
        ]),
        
        # ===== MÉTRICAS DO MODELO =====
        section_header(
            '📈 Métricas e Performance',
            'Avaliação do modelo treinado'
        ),
        
        html.Div(id='model-metrics-container')
        
    ], style={'padding': '20px'})


@callback(
    Output('shap-feature-importance', 'figure'),
    Input('sample-index-slider', 'value'),
    prevent_initial_call=False
)
def update_shap_feature_importance(sample_idx):
    """Update SHAP feature importance bar plot."""
    ctx = get_context()
    clf = ctx.classifier
    
    if clf.shap_values is None or clf.feature_names is None:
        return go.Figure().add_annotation(
            text='SHAP values não disponíveis',
            showarrow=False
        )
    
    cache_key = _shap_cache_key(ctx, 'feature-importance')
    if cache_key in _SHAP_FIGURE_CACHE:
        return _SHAP_FIGURE_CACHE[cache_key]

    try:
        fig = create_shap_feature_importance_bar(
            clf.shap_values,
            clf.feature_names,
            class_names=list(clf.label_encoder.classes_),
            top_n=15,
            title='🎯 Feature Importance - Média Absoluta |SHAP|'
        )
        _SHAP_FIGURE_CACHE[cache_key] = fig
        return fig
    except Exception as e:
        return go.Figure().add_annotation(
            text=f'Erro ao gerar gráfico: {str(e)}',
            showarrow=False
        )


@callback(
    Output('shap-multiclass-bar', 'figure'),
    Input('sample-index-slider', 'value'),
    prevent_initial_call=False
)
def update_shap_multiclass_bar(sample_idx):
    """Update SHAP multiclass bar plot."""
    ctx = get_context()
    clf = ctx.classifier
    
    if clf.shap_values is None or clf.feature_names is None:
        return go.Figure().add_annotation(
            text='SHAP values não disponíveis',
            showarrow=False
        )
    
    # Verificar se é multiclasse
    if not isinstance(clf.shap_values, list) or len(clf.shap_values) <= 1:
        return go.Figure().add_annotation(
            text='Gráfico disponível apenas para problemas multiclasse',
            showarrow=False
        )
    
    cache_key = _shap_cache_key(ctx, 'multiclass-bar')
    if cache_key in _SHAP_FIGURE_CACHE:
        return _SHAP_FIGURE_CACHE[cache_key]

    try:
        fig = create_shap_multiclass_bar(
            clf.shap_values,
            clf.feature_names,
            list(clf.label_encoder.classes_),
            top_n=10,
            title='📊 Feature Importance por Classe (SHAP)'
        )
        _SHAP_FIGURE_CACHE[cache_key] = fig
        return fig
    except Exception as e:
        return go.Figure().add_annotation(
            text=f'Erro ao gerar gráfico: {str(e)}',
            showarrow=False
        )


@callback(
    Output('shap-beeswarm', 'figure'),
    Input('sample-index-slider', 'value'),
    prevent_initial_call=False
)
def update_shap_beeswarm(sample_idx):
    """Update SHAP beeswarm plot."""
    ctx = get_context()
    clf = ctx.classifier
    
    if clf.shap_values is None or clf.shap_data is None or clf.feature_names is None:
        return go.Figure().add_annotation(
            text='SHAP data não disponível',
            showarrow=False
        )
    
    cache_key = _shap_cache_key(ctx, 'beeswarm')
    if cache_key in _SHAP_FIGURE_CACHE:
        return _SHAP_FIGURE_CACHE[cache_key]

    try:
        # Mostrar apenas as 3 features climáticas + dados demográficos (Idade, Gênero)
        include = ['Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)', 'Idade', 'Gênero']
        fig = create_shap_beeswarm(
            clf.shap_values,
            clf.shap_data,
            clf.feature_names,
            top_n=15,
            title='🐝 SHAP Beeswarm Plot - Climáticas + Demográficas',
            include_features=include
        )
        _SHAP_FIGURE_CACHE[cache_key] = fig
        return fig
    except Exception as e:
        return go.Figure().add_annotation(
            text=f'Erro ao gerar gráfico: {str(e)}',
            showarrow=False
        )


@callback(
    [
        Output('shap-force-plot', 'figure'),
        Output('sample-index-display', 'children')
    ],
    Input('sample-index-slider', 'value'),
    prevent_initial_call=True
)
def update_shap_force_plot(sample_idx):
    """Update SHAP force plot for selected sample."""
    ctx = get_context()
    clf = ctx.classifier
    
    display_text = f'Amostra #{sample_idx} (de 0-99)'
    
    if clf.shap_values is None or clf.shap_data is None or clf.feature_names is None:
        return (
            go.Figure().add_annotation(text='SHAP data não disponível', showarrow=False),
            display_text
        )
    
    cache_key = _shap_cache_key(ctx, 'force-plot', sample_idx)
    if cache_key in _SHAP_FIGURE_CACHE:
        return _SHAP_FIGURE_CACHE[cache_key], display_text

    try:
        # Pegar predição para a amostra
        X_sample = clf.shap_data[sample_idx:sample_idx+1]
        pred = clf.predict(X_sample)[0]
        pred_proba = clf.predict_proba(X_sample)[0]
        predicted_class = clf.label_encoder.classes_[pred]
        confidence = pred_proba[pred]
        
        # Determinar classe_idx para multiclasse
        class_idx = 0 if not isinstance(clf.shap_values, list) else pred
        
        # Base value (média de predições)
        if isinstance(clf.shap_values, list):
            base_value = 0.5  # Aproximação
        else:
            base_value = 0.5
        
        fig = create_shap_force_plot(
            clf.shap_values,
            base_value,
            clf.shap_data[sample_idx],
            clf.feature_names,
            predicted_class,
            sample_idx=sample_idx,
            class_idx=class_idx,
            max_display=10
        )
        
        # Adicionar informações de confiança
        display_text = f'Amostra #{sample_idx} | 🎯 Predição: {predicted_class} (confiança: {confidence:.1%})'
        
        _SHAP_FIGURE_CACHE[cache_key] = fig
        return fig, display_text
        
    except Exception as e:
        import traceback
        return (
            go.Figure().add_annotation(text=f'Erro: {str(e)}', showarrow=False),
            f'Amostra #{sample_idx} - Erro: {str(e)}'
        )


@callback(
    Output('model-metrics-container', 'children'),
    Input('sample-index-slider', 'value'),
    prevent_initial_call=False
)
def update_model_metrics(sample_idx):
    """Update model metrics display."""
    ctx = get_context()
    clf = ctx.classifier
    
    if clf.metrics is None:
        return html.Div('Métricas não disponíveis')
    
    metrics = clf.metrics
    best_params = clf.best_params
    
    # Criar cards de métricas
    metric_cards = [
        dbc.Col([
            html.Div([
                html.H6('Acurácia', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em', 'margin': '0'}),
                html.H4(f"{metrics['accuracy']:.1%}", style={'color': COLORS['primary'], 'margin': '8px 0 0 0', 'fontWeight': '700'})
            ], style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                'padding': '16px',
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["border"]}'
            })
        ], md=3, sm=6),
        dbc.Col([
            html.Div([
                html.H6('Acurácia Balanceada', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em', 'margin': '0'}),
                html.H4(f"{metrics['balanced_accuracy']:.1%}", style={'color': COLORS['success'], 'margin': '8px 0 0 0', 'fontWeight': '700'})
            ], style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                'padding': '16px',
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["border"]}'
            })
        ], md=3, sm=6),
        dbc.Col([
            html.Div([
                html.H6('Precision (Macro)', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em', 'margin': '0'}),
                html.H4(f"{metrics['precision_macro']:.1%}", style={'color': COLORS['warning'], 'margin': '8px 0 0 0', 'fontWeight': '700'})
            ], style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                'padding': '16px',
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["border"]}'
            })
        ], md=3, sm=6),
        dbc.Col([
            html.Div([
                html.H6('F1-Score (Macro)', style={'color': COLORS['text_secondary'], 'fontSize': '0.85em', 'margin': '0'}),
                html.H4(f"{metrics['f1_macro']:.1%}", style={'color': COLORS['accent'], 'margin': '8px 0 0 0', 'fontWeight': '700'})
            ], style={
                'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
                'padding': '16px',
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["border"]}'
            })
        ], md=3, sm=6),
    ]
    
    # Adicionar melhor score de tuning se disponível
    children = [dbc.Row(metric_cards, className='mb-4')]
    
    if best_params:
        children.append(html.Div([
            html.H5('🔧 Melhores Parâmetros (Tuning)', style={'color': COLORS['text'], 'marginBottom': '12px'}),
            html.Pre(
                '\n'.join([f'{k}: {v}' for k, v in best_params.items()]),
                style={
                    'backgroundColor': COLORS['secondary'],
                    'color': COLORS['text_secondary'],
                    'padding': '12px',
                    'borderRadius': '6px',
                    'fontSize': '0.9em',
                    'border': f'1px solid {COLORS["border"]}'
                }
            )
        ], style={'marginTop': '20px'}))
    
    return children


def register_callbacks(app):
    """Register classification callbacks with the app.
    
    Note: Callbacks are registered using @callback decorator,
    so this function is just a placeholder for consistency.
    """   
    pass
