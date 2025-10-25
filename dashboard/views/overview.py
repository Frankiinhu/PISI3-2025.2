"""Overview tab layout and callbacks."""
from __future__ import annotations

from dash import Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        univariate_header,
        create_card([dcc.Graph(id='diagnosis-count-graph')], 'Distribuição de Diagnósticos'),
        univariate_top,
        create_card([dcc.Graph(id='climate-vars-distribution')], 'Distribuição de Variáveis Climáticas'),
    ])


def register_callbacks(app) -> None:
    @app.callback(Output('diagnosis-count-graph', 'figure'), Input('tabs', 'value'))
    def update_diagnosis_count(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
        diag_counts = ctx.df[diagnosis_col].value_counts().reset_index()
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

    @app.callback(Output('age-dist-univariate', 'figure'), Input('tabs', 'value'))
    def update_age_distribution(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        fig = px.histogram(ctx.df, x='Idade', nbins=30, color_discrete_sequence=[COLORS['primary']])

        mean_age = ctx.df['Idade'].mean()
        median_age = ctx.df['Idade'].median()
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

    @app.callback(Output('gender-dist-univariate', 'figure'), Input('tabs', 'value'))
    def update_gender_distribution(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        gender_counts = ctx.df['Gênero'].value_counts().reset_index()
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

    @app.callback(Output('temp-dist-univariate', 'figure'), Input('tabs', 'value'))
    def update_temp_distribution(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        fig = px.histogram(ctx.df, x='Temperatura (°C)', nbins=30, color_discrete_sequence=[COLORS['accent']])

        mean_temp = ctx.df['Temperatura (°C)'].mean()
        fig.add_vline(x=mean_temp, line_dash='dash', line_color=COLORS['primary'], annotation_text=f'Média: {mean_temp:.1f}°C')

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Temperatura (°C)',
            yaxis_title='Frequência',
            showlegend=False,
            xaxis=dict(gridcolor=COLORS['border']),
            yaxis=dict(gridcolor=COLORS['border']),
        )
        return fig

    @app.callback(Output('humidity-dist-univariate', 'figure'), Input('tabs', 'value'))
    def update_humidity_distribution(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        fig = px.histogram(ctx.df, x='Umidade', nbins=30, color_discrete_sequence=[COLORS['primary']])

        mean_humidity = ctx.df['Umidade'].mean()
        fig.add_vline(x=mean_humidity, line_dash='dash', line_color=COLORS['accent'], annotation_text=f'Média: {mean_humidity:.2f}')

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Umidade',
            yaxis_title='Frequência',
            showlegend=False,
            xaxis=dict(gridcolor=COLORS['border']),
            yaxis=dict(gridcolor=COLORS['border']),
        )
        return fig

    @app.callback(Output('wind-dist-univariate', 'figure'), Input('tabs', 'value'))
    def update_wind_distribution(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        fig = px.histogram(
            ctx.df,
            x='Velocidade do Vento (km/h)',
            nbins=30,
            color_discrete_sequence=[COLORS['accent_secondary']],
        )

        mean_wind = ctx.df['Velocidade do Vento (km/h)'].mean()
        fig.add_vline(x=mean_wind, line_dash='dash', line_color=COLORS['primary'], annotation_text=f'Média: {mean_wind:.1f} km/h')

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            xaxis_title='Velocidade do Vento (km/h)',
            yaxis_title='Frequência',
            showlegend=False,
            xaxis=dict(gridcolor=COLORS['border']),
            yaxis=dict(gridcolor=COLORS['border']),
        )
        return fig

    @app.callback(Output('climate-vars-distribution', 'figure'), Input('tabs', 'value'))
    def update_climate_distribution(tab):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        if not ctx.climatic_vars:
            return go.Figure()

        fig = make_subplots(rows=len(ctx.climatic_vars), cols=1, subplot_titles=ctx.climatic_vars)
        colors = [COLORS['primary'], COLORS['accent'], '#FF6B6B']

        for idx, var in enumerate(ctx.climatic_vars, 1):
            fig.add_trace(
                go.Histogram(x=ctx.df[var], name=var, marker_color=colors[(idx - 1) % len(colors)]),
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
