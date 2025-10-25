"""Exploratory Analysis tab layout and callbacks."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components import create_card
from ..core.data_context import get_context, get_cluster_feature_frame, has_feature_importances, is_classifier_available
from ..core.theme import COLORS, page_header


def _section_header(title: str, subtitle: str | None = None, accent: str = 'accent') -> html.Div:
    return html.Div([
        html.H3(
            title,
            style={
                'color': COLORS['text'],
                'marginBottom': '10px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS[accent]}',
                'paddingLeft': '14px',
                'background': f'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%)',
            },
        ),
        html.P(
            subtitle,
            style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '24px',
                'paddingLeft': '18px',
            },
        ) if subtitle else None,
    ])


def _graph_row(cards: Sequence[html.Div]) -> html.Div:
    return html.Div(cards, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '20px'})


def _graph_card(graph_id: str, title: str, flex: str = '1 1 360px') -> html.Div:
    return html.Div(create_card([dcc.Graph(id=graph_id)], title), style={'flex': flex, 'minWidth': '320px'})


def _filter_dropdown(component_id: str, label: str, options: Iterable[dict], value, width: str = '25%') -> html.Div:
    return html.Div([
        html.Label(label, style={'color': COLORS['text'], 'fontWeight': '600', 'display': 'block', 'marginBottom': '8px'}),
        dcc.Dropdown(
            id=component_id,
            options=list(options),
            value=value,
            clearable=False,
            className='custom-dropdown',
            style={'backgroundColor': COLORS['secondary']},
        ),
    ], style={'flex': f'1 1 {width}', 'minWidth': '220px'})


def create_layout() -> html.Div:
    ctx = get_context()
    diagnosis_label = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
    clean_symptoms = [s for s in ctx.symptom_cols if 'HIV' not in s.upper() and 'AIDS' not in s.upper()]
    symptom_options = [{'label': s, 'value': s} for s in clean_symptoms][:20]
    default_symptoms = [opt['value'] for opt in symptom_options][:4]

    climate_filter_options = [
        (
            'temp-profile-filter',
            '🌡️ Temperatura',
            [
                {'label': '🔥 Alto (>25°C)', 'value': 'alto'},
                {'label': '🌤️ Médio (18-25°C)', 'value': 'medio'},
                {'label': '❄️ Baixo (<18°C)', 'value': 'baixo'},
                {'label': '✨ Todos', 'value': 'todos'},
            ],
            'todos',
        ),
        (
            'humidity-profile-filter',
            '💧 Umidade',
            [
                {'label': '💦 Alta (>0.7)', 'value': 'alto'},
                {'label': '💧 Média (0.4-0.7)', 'value': 'medio'},
                {'label': '🏜️ Baixa (<0.4)', 'value': 'baixo'},
                {'label': '✨ Todas', 'value': 'todos'},
            ],
            'todos',
        ),
        (
            'wind-profile-filter',
            '💨 Vento',
            [
                {'label': '🌪️ Alto (>15 km/h)', 'value': 'alto'},
                {'label': '🍃 Médio (5-15 km/h)', 'value': 'medio'},
                {'label': '🌿 Baixo (<5 km/h)', 'value': 'baixo'},
                {'label': '✨ Todos', 'value': 'todos'},
            ],
            'todos',
        ),
        (
            'view-type-filter',
            '📊 Visualizar',
            [
                {'label': '🩺 Diagnósticos', 'value': 'diagnosticos'},
                {'label': '💊 Sintomas (Top 10)', 'value': 'sintomas'},
            ],
            'diagnosticos',
        ),
    ]

    demographic_filters = [
        (
            'gender-filter',
            '👤 Gênero',
            [
                {'label': '👨 Masculino', 'value': 1},
                {'label': '👩 Feminino', 'value': 0},
                {'label': '✨ Todos', 'value': 'todos'},
            ],
            'todos',
        ),
        (
            'age-filter',
            '🎂 Faixa Etária',
            [
                {'label': '👶 Crianças (0-12)', 'value': 'crianca'},
                {'label': '🧒 Adolescentes (13-17)', 'value': 'adolescente'},
                {'label': '👨 Adultos (18-59)', 'value': 'adulto'},
                {'label': '👴 Idosos (60+)', 'value': 'idoso'},
                {'label': '✨ Todos', 'value': 'todos'},
            ],
            'todos',
        ),
    ]

    return html.Div([
        page_header(
            'Análise Exploratória de Dados',
            'Explore correlações, padrões e incidências entre variáveis climáticas, demográficas e sintomas.',
            '',
        ),
        _section_header('🔗 Análise Bivariada', 'Relações entre variáveis climáticas e diagnósticos.'),
        _graph_row([
            _graph_card('temp-diagnosis-graph', 'Temperatura vs Diagnóstico'),
            _graph_card('humidity-diagnosis-graph', 'Umidade vs Diagnóstico'),
            _graph_card('wind-diagnosis-graph', 'Velocidade do Vento vs Diagnóstico'),
        ]),
        _graph_row([
            _graph_card('symptom-diagnosis-correlation', 'Matriz Sintomas x Diagnósticos', flex='1 1 600px'),
            _graph_card('correlation-matrix-graph', 'Matriz de Correlação (Top Features)', flex='1 1 600px'),
        ]),
        _graph_row([
            _graph_card('age-temp-distribution', 'Distribuição Etária por Faixa Climática'),
            _graph_card('wind-respiratory-scatter', 'Regressão: Vento vs Sintomas Respiratórios'),
        ]),
        _section_header('🩺 Análise de Sintomas', 'Entenda a incidência de sintomas e diagnósticos associados.', accent='accent_secondary'),
        html.Div([
            html.Label('Selecione Sintomas para Análise:', style={'color': COLORS['text'], 'fontWeight': '600', 'marginBottom': '12px', 'display': 'block'}),
            dcc.Dropdown(
                id='symptom-selector',
                options=symptom_options,
                value=default_symptoms,
                multi=True,
                placeholder='Selecione sintomas...',
                className='custom-dropdown',
                style={'backgroundColor': COLORS['secondary'], 'borderRadius': '8px'},
            ),
        ], style={
            'marginBottom': '25px',
            'padding': '24px',
            'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
            'borderRadius': '14px',
            'boxShadow': '0 8px 32px rgba(0,0,0,0.35)',
            'border': f'1px solid {COLORS["border"]}',
        }),
        _graph_row([
            _graph_card('symptom-frequency-graphs', 'Frequência de Sintomas por Diagnóstico', flex='1 1 600px'),
        ]),
        _graph_row([
            _graph_card('diagnosis-by-symptom-graph', 'Diagnósticos por Sintoma (Top 10)'),
            _graph_card('symptom-importance-graph', 'Importância dos Sintomas (Modelo)'),
        ]),
        _section_header('🌤️ Explorador Climático Interativo', 'Filtre condições e observe impactos em sintomas e diagnósticos.'),
        html.Div([
            _graph_row([
                _filter_dropdown(filter_id, label, options, value) for filter_id, label, options, value in climate_filter_options
            ]),
            _graph_row([
                _filter_dropdown(filter_id, label, options, value, width='40%')
                for filter_id, label, options, value in demographic_filters
            ]),
            html.Div(id='filter-stats', style={
                'marginTop': '16px',
                'padding': '16px',
                'backgroundColor': COLORS['background'],
                'borderRadius': '10px',
                'borderLeft': f'4px solid {COLORS["accent"]}',
            }),
        ], style={
            'padding': '24px',
            'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
            'borderRadius': '14px',
            'boxShadow': '0 8px 32px rgba(0,0,0,0.35)',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
        }),
        _graph_row([
            _graph_card('climate-explorer-graph', 'Incidência no Perfil Selecionado', flex='1 1 600px'),
            _graph_card('climate-correlation-graph', 'Correlação com Condições Climáticas', flex='1 1 600px'),
        ]),
    ])


def register_callbacks(app) -> None:
    diagnosis_col = lambda: (get_context().diagnosis_cols[0] if get_context().diagnosis_cols else 'Diagnóstico')

    @app.callback(Output('symptom-frequency-graphs', 'figure'), [Input('symptom-selector', 'value'), Input('tabs', 'value')])
    def update_symptom_frequency(selected_symptoms, tab):
        if tab != 'tab-eda' or not selected_symptoms:
            return go.Figure()

        ctx = get_context()
        rows = (len(selected_symptoms) + 1) // 2
        fig = make_subplots(rows=rows, cols=2, subplot_titles=selected_symptoms, vertical_spacing=0.16, horizontal_spacing=0.1)

        for idx, symptom in enumerate(selected_symptoms):
            if symptom not in ctx.df.columns:
                continue
            freq = ctx.df.groupby(diagnosis_col())[symptom].sum().reset_index()
            freq.columns = ['Diagnóstico', 'Contagem']
            row, col = divmod(idx, 2)
            fig.add_trace(
                go.Bar(x=freq['Diagnóstico'], y=freq['Contagem'], marker_color=COLORS['accent'], showlegend=False),
                row=row + 1,
                col=col + 1,
            )

        fig.update_layout(
            height=max(320, 320 * rows),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
        )
        fig.update_xaxes(tickangle=-45)
        return fig

    @app.callback(Output('correlation-matrix-graph', 'figure'), Input('tabs', 'value'))
    def update_correlation_matrix(tab):
        if tab != 'tab-eda':
            return go.Figure()

        ctx = get_context()
        base_features = ['Idade', 'Gênero', 'Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)']
        available_base = [feature for feature in base_features if feature in ctx.df.columns]

        top_additional: list[str] = []
        if has_feature_importances():
            feature_importances = getattr(ctx.classifier, 'feature_importances', None)
            if feature_importances is not None:
                filtered = feature_importances[~feature_importances.index.isin(base_features)]
                top_additional = filtered.head(10).index.tolist()
        if not top_additional:
            filtered_symptoms = [col for col in ctx.symptom_cols if 'HIV' not in col.upper() and 'AIDS' not in col.upper()]
            if filtered_symptoms:
                symptom_sums = ctx.df[filtered_symptoms].sum().sort_values(ascending=False)
                top_additional = symptom_sums.head(10).index.tolist()

        features = available_base + top_additional
        if not features:
            return go.Figure()

        corr = ctx.df[features].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={'size': 8},
                colorbar=dict(title='Correlação'),
            )
        )
        fig.update_layout(
            height=600,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
        )
        fig.update_xaxes(tickangle=-45)
        return fig

    @app.callback(Output('age-temp-distribution', 'figure'), Input('tabs', 'value'))
    def update_age_temp_distribution(tab):
        if tab != 'tab-eda':
            return go.Figure()

        ctx = get_context()
        df_temp = ctx.df.copy()
        df_temp['Faixa Temperatura'] = pd.cut(
            df_temp['Temperatura (°C)'],
            bins=[0, 15, 20, 25, 30, np.inf],
            labels=['<15°C', '15-20°C', '20-25°C', '25-30°C', '>30°C'],
        )
        df_temp['Faixa Etária'] = pd.cut(
            df_temp['Idade'],
            bins=[0, 18, 30, 45, 60, np.inf],
            labels=['0-18', '19-30', '31-45', '46-60', '60+'],
        )
        distribution = df_temp.groupby(['Faixa Temperatura', 'Faixa Etária']).size().reset_index(name='Contagem')

        fig = go.Figure()
        palette = [COLORS['primary'], COLORS['primary_light'], COLORS['accent'], COLORS['accent_secondary'], COLORS['secondary']]
        for idx, faixa in enumerate(['0-18', '19-30', '31-45', '46-60', '60+']):
            subset = distribution[distribution['Faixa Etária'] == faixa]
            fig.add_bar(
                x=subset['Faixa Temperatura'],
                y=subset['Contagem'],
                name=faixa,
                marker_color=palette[idx % len(palette)],
            )

        fig.update_layout(
            barmode='group',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Faixa de Temperatura',
            yaxis_title='Número de Pacientes',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        return fig

    @app.callback(Output('wind-respiratory-scatter', 'figure'), Input('tabs', 'value'))
    def update_wind_respiratory_scatter(tab):
        if tab != 'tab-eda':
            return go.Figure()

        ctx = get_context()
        respiratory_symptoms = [
            'Coriza', 'Tosse', 'Dor de Garganta', 'Congestão Nasal', 'Dificuldade Respiratória', 'Chiado no Peito'
        ]
        available = [sym for sym in respiratory_symptoms if sym in ctx.df.columns]
        if not available:
            fig = go.Figure()
            fig.add_annotation(
                text='Sintomas respiratórios não encontrados no dataset atual.',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS['text_secondary']),
            )
            fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['card'])
            return fig

        df = ctx.df.copy()
        df['Freq_Respiratórios'] = df[available].sum(axis=1) / len(available)
        df['Wind_Bins'] = pd.cut(df['Velocidade do Vento (km/h)'], bins=20)
        grouped = df.groupby('Wind_Bins').agg({
            'Freq_Respiratórios': 'mean',
            'Velocidade do Vento (km/h)': 'mean',
        }).dropna().reset_index()

        x = grouped['Velocidade do Vento (km/h)'].to_numpy()
        y = grouped['Freq_Respiratórios'].to_numpy()
        coef = np.polyfit(x, y, 1)
        y_pred = np.poly1d(coef)(x)
        r_squared = np.corrcoef(y, y_pred)[0, 1] ** 2 if len(x) > 1 else 0.0

        fig = go.Figure()
        fig.add_scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=10, color=COLORS['primary'], opacity=0.65, line=dict(color='white', width=1)),
            name='Dados',
        )
        fig.add_scatter(
            x=x,
            y=y_pred,
            mode='lines',
            line=dict(color=COLORS['accent'], width=3, dash='dash'),
            name=f'Regressão Linear (R²={r_squared:.3f})',
        )
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Velocidade do Vento (km/h)',
            yaxis_title='Frequência Média de Sintomas',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            annotations=[{
                'text': f'y = {coef[0]:.4f}x + {coef[1]:.4f}',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.02,
                'y': 0.98,
                'showarrow': False,
                'font': {'color': COLORS['text'], 'size': 12},
            }],
        )
        fig.update_yaxes(tickformat='.0%')
        return fig

    def _climate_box_plot(column: str, graph_id: str, color: str) -> None:
        @app.callback(Output(graph_id, 'figure'), Input('tabs', 'value'))
        def _update(tab: str, data_column: str = column, graph_color: str = color):
            if tab != 'tab-eda':
                return go.Figure()
            ctx = get_context()
            if data_column not in ctx.df.columns:
                return go.Figure()
            fig = px.box(
                ctx.df,
                x=diagnosis_col(),
                y=data_column,
                color_discrete_sequence=[graph_color],
            )
            fig.update_layout(
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['card'],
                font_color=COLORS['text'],
                showlegend=False,
                xaxis_tickangle=-45,
                xaxis=dict(gridcolor=COLORS['border']),
                yaxis=dict(gridcolor=COLORS['border']),
            )
            return fig

    _climate_box_plot('Temperatura (°C)', 'temp-diagnosis-graph', COLORS['accent'])
    _climate_box_plot('Umidade', 'humidity-diagnosis-graph', COLORS['primary'])
    _climate_box_plot('Velocidade do Vento (km/h)', 'wind-diagnosis-graph', COLORS['accent_secondary'])

    @app.callback(Output('symptom-diagnosis-correlation', 'figure'), Input('tabs', 'value'))
    def update_symptom_diagnosis_correlation(tab):
        if tab != 'tab-eda':
            return go.Figure()

        ctx = get_context()
        filtered_symptoms = [col for col in ctx.symptom_cols if 'HIV' not in col.upper() and 'AIDS' not in col.upper()]
        if not filtered_symptoms:
            return go.Figure()

        top_symptoms = ctx.df[filtered_symptoms].sum().sort_values(ascending=False).head(20).index.tolist()
        diagnoses = sorted(ctx.df[diagnosis_col()].unique())
        matrix = []
        for diag in diagnoses:
            subset = ctx.df[ctx.df[diagnosis_col()] == diag]
            proportions = [subset[symptom].mean() if symptom in subset.columns else 0 for symptom in top_symptoms]
            matrix.append(proportions)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[sym.replace('_', ' ').title() for sym in top_symptoms],
                y=diagnoses,
                colorscale='Blues',
                zmin=0,
                zmax=1,
                colorbar=dict(title='Proporção'),
                hovertemplate='<b>Diagnóstico:</b> %{y}<br><b>Sintoma:</b> %{x}<br><b>Proporção:</b> %{z:.1%}<extra></extra>',
            )
        )
        fig.update_layout(
            height=600,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            margin=dict(l=150, r=80, t=60, b=140),
        )
        fig.update_xaxes(tickangle=-45)
        return fig

    @app.callback(Output('diagnosis-by-symptom-graph', 'figure'), Input('tabs', 'value'))
    def update_diagnosis_by_symptom(tab):
        if tab != 'tab-eda':
            return go.Figure()

        ctx = get_context()
        filtered_symptoms = [col for col in ctx.symptom_cols if 'HIV' not in col.upper() and 'AIDS' not in col.upper()]
        if not filtered_symptoms:
            return go.Figure()

        top_symptoms = ctx.df[filtered_symptoms].sum().sort_values(ascending=False).head(10).index.tolist()
        fig = make_subplots(rows=5, cols=2, subplot_titles=top_symptoms, vertical_spacing=0.08, horizontal_spacing=0.12)
        palette = [
            '#5559ff', '#7b7fff', '#a4a8ff', '#4facfe', '#00c9a7',
            '#fbbf24', '#f87171', '#4ade80', '#60a5fa', '#a78bfa',
        ]
        for idx, symptom in enumerate(top_symptoms):
            subset = ctx.df[ctx.df[symptom] == 1]
            counts = subset[diagnosis_col()].value_counts()
            row, col = divmod(idx, 2)
            fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker=dict(color=palette[idx % len(palette)], line=dict(color=COLORS['border'], width=1)),
                    showlegend=False,
                ),
                row=row + 1,
                col=col + 1,
            )
            fig.update_xaxes(tickangle=-45, row=row + 1, col=col + 1, gridcolor=COLORS['border'])
            fig.update_yaxes(row=row + 1, col=col + 1, gridcolor=COLORS['border'])

        fig.update_layout(
            height=1400,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font=dict(family='Inter, sans-serif', color=COLORS['text']),
            margin=dict(t=80, b=60, l=60, r=40),
        )
        return fig

    @app.callback(Output('symptom-importance-graph', 'figure'), Input('tabs', 'value'))
    def update_symptom_importance(tab):
        if tab != 'tab-eda' or not is_classifier_available():
            return go.Figure()

        ctx = get_context()
        feature_importances = getattr(ctx.classifier, 'feature_importances', None)
        if feature_importances is None:
            return go.Figure()

        top_features = feature_importances.head(15)
        fig = px.bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            color=top_features.values,
            color_continuous_scale='Blues',
        )
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Importância',
            yaxis_title='Feature',
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
        )
        return fig

    @app.callback(
        [Output('filter-stats', 'children'), Output('climate-explorer-graph', 'figure'), Output('climate-correlation-graph', 'figure')],
        [
            Input('temp-profile-filter', 'value'),
            Input('humidity-profile-filter', 'value'),
            Input('wind-profile-filter', 'value'),
            Input('gender-filter', 'value'),
            Input('age-filter', 'value'),
            Input('view-type-filter', 'value'),
            Input('tabs', 'value'),
        ],
    )
    def update_climate_explorer(temp_profile, humidity_profile, wind_profile, gender, age_group, view_type, tab):
        if tab != 'tab-eda':
            return html.Div(), go.Figure(), go.Figure()

        ctx = get_context()
        df_filtered = ctx.df.copy()

        if temp_profile == 'alto':
            df_filtered = df_filtered[df_filtered['Temperatura (°C)'] > 25]
        elif temp_profile == 'medio':
            df_filtered = df_filtered[df_filtered['Temperatura (°C)'].between(18, 25)]
        elif temp_profile == 'baixo':
            df_filtered = df_filtered[df_filtered['Temperatura (°C)'] < 18]

        if humidity_profile == 'alto':
            df_filtered = df_filtered[df_filtered['Umidade'] > 0.7]
        elif humidity_profile == 'medio':
            df_filtered = df_filtered[df_filtered['Umidade'].between(0.4, 0.7)]
        elif humidity_profile == 'baixo':
            df_filtered = df_filtered[df_filtered['Umidade'] < 0.4]

        if wind_profile == 'alto':
            df_filtered = df_filtered[df_filtered['Velocidade do Vento (km/h)'] > 15]
        elif wind_profile == 'medio':
            df_filtered = df_filtered[df_filtered['Velocidade do Vento (km/h)'].between(5, 15)]
        elif wind_profile == 'baixo':
            df_filtered = df_filtered[df_filtered['Velocidade do Vento (km/h)'] < 5]

        if gender != 'todos':
            df_filtered = df_filtered[df_filtered['Gênero'] == gender]

        if age_group == 'crianca':
            df_filtered = df_filtered[df_filtered['Idade'] <= 12]
        elif age_group == 'adolescente':
            df_filtered = df_filtered[df_filtered['Idade'].between(13, 17)]
        elif age_group == 'adulto':
            df_filtered = df_filtered[df_filtered['Idade'].between(18, 59)]
        elif age_group == 'idoso':
            df_filtered = df_filtered[df_filtered['Idade'] >= 60]

        total_original = len(ctx.df)
        total_filtered = len(df_filtered)
        percent = (total_filtered / total_original * 100) if total_original else 0

        stats_div = html.Div([
            html.Div([
                html.Span('📊 Registros: ', style={'fontWeight': '600', 'color': COLORS['text']}),
                html.Span(f'{total_filtered:,} / {total_original:,} ', style={'color': COLORS['accent'], 'fontSize': '1.1em', 'fontWeight': '700'}),
                html.Span(f'({percent:.1f}%)', style={'color': COLORS['text_secondary']}),
            ], style={'marginBottom': '6px'}),
            html.Div([
                html.Span('🔍 Filtros ativos: ', style={'fontWeight': '600', 'color': COLORS['text']}),
                html.Span(', '.join([
                    label for label, condition in [
                        ('🔥 Temperatura Alta', temp_profile == 'alto'),
                        ('🌤️ Temperatura Média', temp_profile == 'medio'),
                        ('❄️ Temperatura Baixa', temp_profile == 'baixo'),
                        ('💦 Umidade Alta', humidity_profile == 'alto'),
                        ('💧 Umidade Média', humidity_profile == 'medio'),
                        ('🏜️ Umidade Baixa', humidity_profile == 'baixo'),
                        ('🌪️ Vento Alto', wind_profile == 'alto'),
                        ('🍃 Vento Médio', wind_profile == 'medio'),
                        ('🌿 Vento Baixo', wind_profile == 'baixo'),
                        ('👨 Masculino', gender == 1),
                        ('👩 Feminino', gender == 0),
                        ('👶 Crianças', age_group == 'crianca'),
                        ('🧒 Adolescentes', age_group == 'adolescente'),
                        ('👨 Adultos', age_group == 'adulto'),
                        ('👴 Idosos', age_group == 'idoso'),
                    ] if condition
                ]) or '✨ Sem filtros aplicados', style={'color': COLORS['accent_secondary']}),
            ]),
        ])

        if total_filtered == 0:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text='Nenhum registro encontrado para os filtros selecionados.',
                xref='paper', yref='paper', x=0.5, y=0.5,
                showarrow=False,
                font=dict(color=COLORS['text_secondary'], size=14),
            )
            empty_fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['card'])
            return stats_div, empty_fig, empty_fig

        if view_type == 'diagnosticos':
            counts = df_filtered[diagnosis_col()].value_counts().reset_index()
            counts.columns = ['Diagnóstico', 'Contagem']
            main_fig = go.Figure()
            main_fig.add_bar(
                x=counts['Diagnóstico'],
                y=counts['Contagem'],
                marker=dict(
                    color=counts['Contagem'],
                    colorscale=[[0, COLORS['primary']], [0.5, COLORS['accent']], [1, COLORS['accent_secondary']]],
                    line=dict(color='white', width=2),
                ),
            )
            main_fig.update_layout(
                title='Incidência de Diagnósticos no Perfil Selecionado',
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['card'],
                font_color=COLORS['text'],
                xaxis=dict(title='Diagnóstico', tickangle=-45, gridcolor=COLORS['border']),
                yaxis=dict(title='Número de Casos', gridcolor=COLORS['border']),
                showlegend=False,
            )
        else:
            filtered_symptoms = [col for col in ctx.symptom_cols if 'HIV' not in col.upper() and 'AIDS' not in col.upper()]
            symptom_totals = df_filtered[filtered_symptoms].sum().sort_values(ascending=False).head(10)
            main_fig = go.Figure()
            main_fig.add_bar(
                y=symptom_totals.index,
                x=symptom_totals.values,
                orientation='h',
                marker=dict(
                    color=symptom_totals.values,
                    colorscale=[[0, COLORS['primary']], [0.5, COLORS['accent']], [1, COLORS['accent_secondary']]],
                    line=dict(color='white', width=2),
                ),
            )
            main_fig.update_layout(
                title='Top 10 Sintomas no Perfil Selecionado',
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['card'],
                font_color=COLORS['text'],
                xaxis=dict(title='Número de Ocorrências', gridcolor=COLORS['border']),
                yaxis=dict(title='Sintoma', gridcolor=COLORS['border']),
                showlegend=False,
            )

        if view_type == 'diagnosticos':
            sample = df_filtered.sample(min(500, len(df_filtered))) if len(df_filtered) > 500 else df_filtered
            corr_fig = go.Figure()
            for diag in sample[diagnosis_col()].unique()[:5]:
                diag_df = sample[sample[diagnosis_col()] == diag]
                corr_fig.add_scatter(
                    x=diag_df['Temperatura (°C)'],
                    y=diag_df['Umidade'],
                    mode='markers',
                    name=diag,
                    marker=dict(size=8, opacity=0.6),
                )
            corr_fig.update_layout(
                title='Temperatura vs Umidade por Diagnóstico',
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['card'],
                font_color=COLORS['text'],
                xaxis=dict(title='Temperatura (°C)', gridcolor=COLORS['border']),
                yaxis=dict(title='Umidade', gridcolor=COLORS['border']),
            )
        else:
            filtered_symptoms = [col for col in ctx.symptom_cols if 'HIV' not in col.upper() and 'AIDS' not in col.upper()][:10]
            temp_bins = pd.cut(df_filtered['Temperatura (°C)'], bins=5)
            avg_symptoms = df_filtered.groupby(temp_bins)[filtered_symptoms].mean().mean(axis=1)
            corr_fig = go.Figure()
            corr_fig.add_scatter(
                x=[str(bin_label) for bin_label in avg_symptoms.index],
                y=avg_symptoms.values,
                mode='lines+markers',
                line=dict(color=COLORS['accent'], width=3),
                marker=dict(size=10, color=COLORS['accent_secondary']),
            )
            corr_fig.update_layout(
                title='Média de Sintomas por Faixa de Temperatura',
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['card'],
                font_color=COLORS['text'],
                xaxis=dict(title='Faixa de Temperatura', tickangle=-45, gridcolor=COLORS['border']),
                yaxis=dict(title='Média de Sintomas', gridcolor=COLORS['border']),
                showlegend=False,
            )

        return stats_div, main_fig, corr_fig


__all__ = ['create_layout', 'register_callbacks']
