"""
Aba de Prognóstico - Predição de diagnósticos com o classificador balanceado (SMOTE).
"""

from dash import dcc, html, callback, Input, Output, State, ALL
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from dashboard.core.data_context import get_context as get_data_context
from dashboard.core.theme import COLORS
from dashboard.components import create_card


def create_prognosis_tab() -> html.Div:
    """Create the prognosis/prediction tab layout."""
    
    return html.Div([
        html.Div([
            html.H2('🔮 Prognóstico - Predição de Diagnóstico', style={
                'color': COLORS['text'],
                'marginBottom': '10px',
                'fontSize': '2.2em',
                'fontWeight': '800',
                'textShadow': f'2px 2px 4px {COLORS["shadow"]}'
            }),
            html.P(
                'Utilize o classificador treinado com SMOTE para prever diagnósticos com base em sintomas, dados demográficos e fatores climáticos.',
                style={
                    'color': COLORS['text_secondary'],
                    'fontSize': '1.1em',
                    'marginBottom': '30px',
                    'lineHeight': '1.6'
                }
            ),
        ], style={'marginBottom': '30px'}),
        
        # Input Section
        html.Div([
            html.H3('📝 Dados do Paciente', style={
                'color': COLORS['text'],
                'marginBottom': '20px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS["primary"]}',
                'paddingLeft': '14px',
            }),
            
            # Symptoms Section
            create_card([
                html.H4('🤒 Sintomas', style={
                    'color': COLORS['text'],
                    'marginBottom': '15px',
                    'fontSize': '1.3em'
                }),
                html.Div(id='prognosis-symptom-inputs', style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fill, minmax(200px, 1fr))',
                    'gap': '15px'
                })
            ], title=''),
            
            # Demographics Section
            html.Div([
                html.Div([
                    create_card([
                        html.H4('👤 Dados Demográficos', style={
                            'color': COLORS['text'],
                            'marginBottom': '15px',
                            'fontSize': '1.3em'
                        }),
                        html.Div([
                            html.Label('Idade:', style={'color': COLORS['text'], 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Input(
                                id='prognosis-age-input',
                                type='number',
                                placeholder='Ex: 45',
                                min=0,
                                max=120,
                                value=30,
                                style={
                                    'width': '100%',
                                    'padding': '10px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORS["border"]}',
                                    'backgroundColor': COLORS['input'],
                                    'color': COLORS['text'],
                                    'fontSize': '1em'
                                }
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Label('Gênero:', style={'color': COLORS['text'], 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='prognosis-gender-dropdown',
                                options=[
                                    {'label': 'Feminino', 'value': 'F'},
                                    {'label': 'Masculino', 'value': 'M'}
                                ],
                                value='F',
                                clearable=False,
                                searchable=False,
                                style={
                                    'width': '100%'
                                }
                            ),
                        ], style={'marginBottom': '15px'})
                    ], title='')
                ], style={'flex': '1', 'minWidth': '250px'}),
                
                # Climate Section
                html.Div([
                    create_card([
                        html.H4('🌤️ Fatores Climáticos', style={
                            'color': COLORS['text'],
                            'marginBottom': '15px',
                            'fontSize': '1.3em'
                        }),
                        html.Div([
                            html.Label('Temperatura (°C):', style={'color': COLORS['text'], 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Input(
                                id='prognosis-temp-input',
                                type='number',
                                placeholder='Ex: 22.5',
                                value=20.0,
                                step=0.1,
                                style={
                                    'width': '100%',
                                    'padding': '10px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORS["border"]}',
                                    'backgroundColor': COLORS['input'],
                                    'color': COLORS['text'],
                                    'fontSize': '1em'
                                }
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Label('Umidade:', style={'color': COLORS['text'], 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Input(
                                id='prognosis-humidity-input',
                                type='number',
                                placeholder='Ex: 0.65',
                                value=0.5,
                                min=0,
                                max=1,
                                step=0.01,
                                style={
                                    'width': '100%',
                                    'padding': '10px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORS["border"]}',
                                    'backgroundColor': COLORS['input'],
                                    'color': COLORS['text'],
                                    'fontSize': '1em'
                                }
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.Label('Velocidade do Vento (km/h):', style={'color': COLORS['text'], 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Input(
                                id='prognosis-wind-input',
                                type='number',
                                placeholder='Ex: 15.0',
                                value=10.0,
                                step=0.1,
                                style={
                                    'width': '100%',
                                    'padding': '10px',
                                    'borderRadius': '8px',
                                    'border': f'2px solid {COLORS["border"]}',
                                    'backgroundColor': COLORS['input'],
                                    'color': COLORS['text'],
                                    'fontSize': '1em'
                                }
                            ),
                        ])
                    ], title='')
                ], style={'flex': '1', 'minWidth': '250px'}),
            ], style={
                'display': 'flex',
                'gap': '20px',
                'flexWrap': 'wrap',
                'marginTop': '20px'
            }),
            
            # Predict Button
            html.Div([
                html.Button(
                    '🔍 Prever Diagnóstico',
                    id='prognosis-predict-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': COLORS['accent'],
                        'color': 'white',
                        'padding': '15px 40px',
                        'fontSize': '1.2em',
                        'fontWeight': 'bold',
                        'border': 'none',
                        'borderRadius': '10px',
                        'cursor': 'pointer',
                        'boxShadow': f'0 4px 6px {COLORS["shadow"]}',
                        'transition': 'all 0.3s ease'
                    }
                )
            ], style={
                'textAlign': 'center',
                'marginTop': '30px',
                'marginBottom': '30px'
            })
        ], style={'marginBottom': '40px'}),
        
        # Results Section
        html.Div([
            html.H3('📊 Resultado da Predição', style={
                'color': COLORS['text'],
                'marginBottom': '20px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS["accent"]}',
                'paddingLeft': '14px',
            }),
            
            html.Div([
                # Predicted Diagnosis Card
                html.Div([
                    create_card([
                        html.Div(id='prognosis-result-card')
                    ], '🎯 Diagnóstico Previsto')
                ], style={'flex': '1', 'minWidth': '300px'}),
                
                # Probability Bar Chart
                html.Div([
                    create_card([
                        dcc.Graph(id='prognosis-probability-chart')
                    ], '📈 Probabilidades por Diagnóstico')
                ], style={'flex': '2', 'minWidth': '400px'}),
            ], style={
                'display': 'flex',
                'gap': '20px',
                'flexWrap': 'wrap'
            })
        ])
    ])


def _get_symptom_columns(ctx) -> List[str]:
    """Extract symptom column names from the dataset."""
    symptom_prefixes = ['Febre', 'Tosse', 'Dor', 'Fadiga', 'Náusea', 'Congestionamento']
    symptom_cols = [col for col in ctx.df.columns 
                    if any(col.startswith(prefix) for prefix in symptom_prefixes)]
    return sorted(symptom_cols)


@callback(
    Output('prognosis-symptom-inputs', 'children'),
    Input('tabs', 'value')
)
def populate_symptom_inputs(tab):
    """Dynamically create symptom checkboxes based on dataset columns."""
    if tab != 'tab-prognosis':
        return []
    
    try:
        ctx = get_data_context()
    except Exception as exc:
        return html.P(f'Erro ao carregar dados: {str(exc)}', style={'color': COLORS['error']})
    
    symptom_cols = _get_symptom_columns(ctx)
    
    if not symptom_cols:
        return html.P('Nenhuma coluna de sintoma encontrada.', style={'color': COLORS['text_secondary']})
    
    checkboxes = []
    for col in symptom_cols:
        checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'prognosis-symptom-check', 'index': col},
                    options=[{'label': col.replace('_', ' '), 'value': col}],
                    value=[],
                    style={'color': COLORS['text']},
                    labelStyle={'display': 'block', 'marginBottom': '5px'}
                )
            ])
        )
    
    return checkboxes


@callback(
    [Output('prognosis-result-card', 'children'),
     Output('prognosis-probability-chart', 'figure')],
    Input('prognosis-predict-button', 'n_clicks'),
    [State('prognosis-age-input', 'value'),
     State('prognosis-gender-dropdown', 'value'),
     State('prognosis-temp-input', 'value'),
     State('prognosis-humidity-input', 'value'),
     State('prognosis-wind-input', 'value'),
     State({'type': 'prognosis-symptom-check', 'index': ALL}, 'value'),
     State({'type': 'prognosis-symptom-check', 'index': ALL}, 'id')]
)
def predict_diagnosis(n_clicks, age, gender, temperature, humidity, wind, symptom_values, symptom_ids):
    """Predict diagnosis using the trained classifier."""
    if n_clicks == 0:
        # Default empty state
        empty_card = html.Div([
            html.P('Clique em "Prever Diagnóstico" para ver o resultado.', 
                   style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'padding': '20px'})
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text'])
        )
        return empty_card, empty_fig
    
    try:
        ctx = get_data_context()
        
        # Check if classifier is available
        if ctx.classifier is None or ctx.classifier.model is None:
            error_card = html.Div([
                html.P('⚠️ Classificador não disponível', 
                       style={'color': COLORS['warning'], 'textAlign': 'center', 'fontSize': '1.2em', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.P('Execute o treinamento do modelo antes de fazer predições.', 
                       style={'color': COLORS['text_secondary'], 'textAlign': 'center'}),
                html.Hr(style={'borderColor': COLORS['border'], 'margin': '15px 0'}),
                html.P([
                    'Execute: ',
                    html.Code('python scripts/train_models.py', 
                             style={'backgroundColor': COLORS['secondary'], 'padding': '4px 8px', 'borderRadius': '4px'})
                ], style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'fontSize': '0.9em'})
            ], style={'padding': '30px'})
            empty_fig = go.Figure()
            empty_fig.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['card'],
                font=dict(color=COLORS['text'])
            )
            return error_card, empty_fig
        
        # Prepare input features
        symptom_cols = _get_symptom_columns(ctx)
        
        # Create feature dict
        features = {}
        
        # Add symptoms (binary)
        for col in symptom_cols:
            features[col] = 0
        
        # Set selected symptoms to 1
        for val, sid in zip(symptom_values, symptom_ids):
            if val:  # If checkbox is checked
                symptom_name = sid['index']
                features[symptom_name] = 1
        
        # Get feature names from trained classifier
        if ctx.classifier.feature_names is None:
            raise ValueError("Classificador não possui feature_names. Execute o treinamento primeiro.")
        
        feature_names = ctx.classifier.feature_names
        
        # Add demographics
        if 'Idade' in feature_names:
            features['Idade'] = age
        
        # Gender encoding (assuming binary: F=0, M=1)
        if 'Gênero_M' in feature_names:
            features['Gênero_M'] = 1 if gender == 'M' else 0
        if 'Gênero_F' in feature_names:
            features['Gênero_F'] = 1 if gender == 'F' else 0
        
        # Add climate
        if 'Temperatura' in feature_names:
            features['Temperatura'] = temperature
        if 'Umidade' in feature_names:
            features['Umidade'] = humidity
        if 'Velocidade do Vento (km/h)' in feature_names:
            features['Velocidade do Vento (km/h)'] = wind
        
        # Create DataFrame with correct feature order order
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Fill missing features with 0
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Check if model is available
        if ctx.classifier.model is None:
            raise ValueError("Modelo não carregado. Execute o treinamento primeiro.")
        
        # Predict
        prediction_encoded = ctx.classifier.model.predict(input_df)[0]
        probabilities = ctx.classifier.model.predict_proba(input_df)[0]
        
        # Decode prediction using label_encoder
        if ctx.classifier.label_encoder is not None:
            prediction = ctx.classifier.label_encoder.inverse_transform([prediction_encoded])[0]
            class_names = ctx.classifier.label_encoder.classes_
        else:
            # Fallback if no label encoder
            prediction = str(prediction_encoded)
            class_names = ctx.classifier.model.classes_
        
        # Result card
        result_card = html.Div([
            html.Div([
                html.H2(prediction, style={
                    'color': COLORS['accent'],
                    'fontSize': '2.5em',
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'textAlign': 'center'
                }),
                html.P(f'Probabilidade: {max(probabilities)*100:.1f}%', style={
                    'color': COLORS['text_secondary'],
                    'fontSize': '1.2em',
                    'textAlign': 'center'
                })
            ])
        ])
        
        # Probability bar chart
        prob_df = pd.DataFrame({
            'Diagnóstico': class_names,
            'Probabilidade': probabilities * 100
        }).sort_values('Probabilidade', ascending=True)
        
        fig = go.Figure()
        
        colors = [COLORS['accent'] if diag == prediction else COLORS['primary'] 
                  for diag in prob_df['Diagnóstico']]
        
        fig.add_trace(go.Bar(
            y=prob_df['Diagnóstico'],
            x=prob_df['Probabilidade'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=2, color='white')),
            hovertemplate='<b>%{y}</b><br>Probabilidade: %{x:.2f}%<extra></extra>',
            text=prob_df['Probabilidade'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ))
        
        fig.update_layout(
            xaxis_title='Probabilidade (%)',
            yaxis_title='Diagnóstico',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text'], family='Inter, sans-serif'),
            height=400,
            showlegend=False,
            xaxis=dict(gridcolor=COLORS['border'], range=[0, 100]),
            yaxis=dict(gridcolor=COLORS['border']),
            margin=dict(l=150)
        )
        
        return result_card, fig
        
    except Exception as e:
        error_card = html.Div([
            html.P(f'Erro ao prever diagnóstico: {str(e)}', 
                   style={'color': 'red', 'textAlign': 'center', 'padding': '20px'})
        ])
        error_fig = go.Figure()
        error_fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(color=COLORS['text'])
        )
        return error_card, error_fig
