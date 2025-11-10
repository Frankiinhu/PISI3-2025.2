"""Overview tab layout and callbacks."""
from __future__ import annotations

from typing import Iterable

from dash import Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ..components import create_card
from ..core.data_context import get_context
from ..core.theme import COLORS, page_header


_STAT_GRID_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
    'gap': '20px',
    'marginBottom': '30px',
}

_STAT_CARD_STYLE = {
    'background': f"linear-gradient(135deg, {COLORS['card']} 0%, {COLORS['card_hover']} 100%)",
    'padding': '28px 20px',
    'borderRadius': '15px',
    'textAlign': 'center',
    'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
    'border': f"1px solid {COLORS['border']}",
    'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
    'cursor': 'pointer',
}

_SECTION_TITLE_STYLE = {
    'color': COLORS['text'],
    'marginBottom': '10px',
    'fontSize': '1.8em',
    'fontWeight': '700',
}

_SECTION_SUBTITLE_STYLE = {
    'color': COLORS['text_secondary'],
    'marginBottom': '25px',
    'fontSize': '1em',
}


def _filter_dropdown(component_id: str, label: str, options: Iterable[dict], value, width: str = '25%') -> html.Div:
    """Helper function to create filter dropdown UI"""
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


def _stat_card(icon: str, label: str, value: str, value_color: str) -> html.Div:
    return html.Div(
        html.Div([
            html.Div(icon, style={'fontSize': '2.5em', 'marginBottom': '10px'}),
            html.H4(label, style={
                'color': COLORS['text_secondary'],
                'margin': '0',
                'fontSize': '0.9em',
                'fontWeight': '500',
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px',
            }),
            html.H2(value, style={
                'color': value_color,
                'margin': '15px 0 0 0',
                'fontSize': '2.5em',
                'fontWeight': '700',
            }),
        ], style=_STAT_CARD_STYLE),
        className='stat-card',
    )


def create_layout() -> html.Div:
    ctx = get_context()
    info = ctx.eda.basic_info()
    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'

    stats_cards = html.Div([
        _stat_card('📊', 'Total de Registros', f"{info['shape'][0]:,}", COLORS['accent']),
        _stat_card('🏥', 'Diagnósticos Únicos', str(ctx.df[diagnosis_col].nunique()), COLORS['success']),
        _stat_card('📈', 'Total de Features', str(info['shape'][1]), COLORS['primary']),
        _stat_card('🔬', 'Sintomas Analisados', str(len(ctx.symptom_cols)), COLORS['warning']),
    ], style=_STAT_GRID_STYLE)

    overview_header = page_header(
        'Visão Geral do Dataset',
        'Estatísticas essenciais e distribuições principais do conjunto de dados',
        '',
    )

    # Controles de filtro para estratificação por gênero
    gender_filter_options = [
        {'label': '👨 Masculino', 'value': 1},
        {'label': '👩 Feminino', 'value': 0},
        {'label': '✨ Todos', 'value': 'todos'},
    ]

    age_filter_options = [
        {'label': '👶 Crianças (0-12)', 'value': 'crianca'},
        {'label': '🧒 Adolescentes (13-17)', 'value': 'adolescente'},
        {'label': '👨 Adultos (18-59)', 'value': 'adulto'},
        {'label': '👴 Idosos (60+)', 'value': 'idoso'},
        {'label': '✨ Todos', 'value': 'todos'},
    ]

    filters_section = html.Div([
        html.H3('🎯 Filtros de Estratificação', style=_SECTION_TITLE_STYLE),
        html.P('Customize a visualização por gênero e faixa etária', style=_SECTION_SUBTITLE_STYLE),
        html.Div([
            _filter_dropdown('overview-gender-filter', '👤 Gênero', gender_filter_options, 'todos', width='40%'),
            _filter_dropdown('overview-age-filter', '🎂 Faixa Etária', age_filter_options, 'todos', width='40%'),
        ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginBottom': '20px', 'padding': '20px', 'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)', 'borderRadius': '14px', 'border': f'1px solid {COLORS["border"]}'}),
    ])

    univariate_header = html.Div([
        html.H3('Análise Univariada', style=_SECTION_TITLE_STYLE),
        html.P('Distribuições individuais das principais variáveis monitoradas.', style=_SECTION_SUBTITLE_STYLE),
    ])

    univariate_top = html.Div([
        html.Div(create_card([dcc.Graph(id='age-dist-univariate')], 'Distribuição de Idade'), style={'flex': '1'}),
        html.Div(create_card([dcc.Graph(id='gender-dist-univariate')], 'Distribuição de Gênero'), style={'flex': '1'}),
    ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginBottom': '20px'})


    return html.Div([
        overview_header,
        stats_cards,
        filters_section,
        create_card([dcc.Graph(id='diagnosis-count-graph')], 'Distribuição de Diagnósticos'),
        univariate_header,
        univariate_top,
        create_card([dcc.Graph(id='climate-vars-distribution')], 'Distribuição de Variáveis Climáticas'),
    ])


def register_callbacks(app) -> None:
    @app.callback(
        Output('diagnosis-count-graph', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-gender-filter', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_diagnosis_count(tab, gender, age_group):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
        
        # Aplicar filtros
        df_filtered = ctx.df.copy()
        
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
        
        diag_counts = df_filtered[diagnosis_col].value_counts().reset_index()
        diag_counts.columns = ['Diagnóstico', 'Contagem']

        fig = px.bar(
            diag_counts,
            x='Diagnóstico',
            y='Contagem',
            color='Contagem',
            color_continuous_scale='Blues',
        )

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            xaxis_title='Diagnóstico',
            yaxis_title='Número de Casos',
            showlegend=False,
            xaxis_tickangle=-45,
            font=dict(family='Inter, sans-serif', size=12),
            title_font=dict(size=16, color=COLORS['text']),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            margin=dict(t=30, b=80, l=60, r=30),
        )
        return fig

    @app.callback(
        Output('age-dist-univariate', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-gender-filter', 'value'),
        ]
    )
    def update_age_distribution(tab, gender):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        df_filtered = ctx.df.copy()
        
        if gender != 'todos':
            df_filtered = df_filtered[df_filtered['Gênero'] == gender]
        
        fig = px.histogram(df_filtered, x='Idade', nbins=30, color_discrete_sequence=[COLORS['primary']])

        mean_age = df_filtered['Idade'].mean()
        median_age = df_filtered['Idade'].median()
        fig.add_vline(x=mean_age, line_dash='dash', line_color=COLORS['accent'], annotation_text=f'Média: {mean_age:.1f}')
        fig.add_vline(x=median_age, line_dash='dot', line_color=COLORS['accent_secondary'], annotation_text=f'Mediana: {median_age:.1f}')

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Idade (anos)',
            yaxis_title='Frequência',
            showlegend=False,
            xaxis=dict(gridcolor=COLORS['border']),
            yaxis=dict(gridcolor=COLORS['border']),
        )
        return fig

    @app.callback(
        Output('gender-dist-univariate', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_gender_distribution(tab, age_group):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        df_filtered = ctx.df.copy()
        
        if age_group == 'crianca':
            df_filtered = df_filtered[df_filtered['Idade'] <= 12]
        elif age_group == 'adolescente':
            df_filtered = df_filtered[df_filtered['Idade'].between(13, 17)]
        elif age_group == 'adulto':
            df_filtered = df_filtered[df_filtered['Idade'].between(18, 59)]
        elif age_group == 'idoso':
            df_filtered = df_filtered[df_filtered['Idade'] >= 60]
        
        gender_counts = df_filtered['Gênero'].value_counts().reset_index()
        gender_counts.columns = ['Gênero', 'Contagem']
        gender_counts['Gênero'] = gender_counts['Gênero'].map({0: 'Feminino', 1: 'Masculino'})

        fig = px.bar(
            gender_counts,
            x='Gênero',
            y='Contagem',
            color='Gênero',
            color_discrete_map={'Feminino': COLORS['accent'], 'Masculino': COLORS['primary']},
        )

        fig.update_traces(texttemplate='%{y}', textposition='outside', showlegend=False)
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Gênero',
            yaxis_title='Contagem',
            xaxis=dict(gridcolor=COLORS['border']),
            yaxis=dict(gridcolor=COLORS['border']),
        )
        return fig

    @app.callback(
        Output('climate-vars-distribution', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-gender-filter', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_climate_distribution(tab, gender, age_group):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        df_filtered = ctx.df.copy()
        
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
        
        if not ctx.climatic_vars:
            return go.Figure()

        fig = make_subplots(rows=len(ctx.climatic_vars), cols=1, subplot_titles=ctx.climatic_vars)
        colors = [COLORS['primary'], COLORS['accent'], '#FF6B6B']

        for idx, var in enumerate(ctx.climatic_vars, 1):
            fig.add_trace(
                go.Histogram(x=df_filtered[var], name=var, marker_color=colors[(idx - 1) % len(colors)]),
                row=idx,
                col=1,
            )

        fig.update_layout(
            height=220 * len(ctx.climatic_vars),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            showlegend=False,
        )
        return fig


__all__ = ['create_layout', 'register_callbacks']
