"""Overview tab layout and callbacks."""
from __future__ import annotations



from dash import Input, Output, dcc, html, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dash_bootstrap_components as dbc

from ..components import create_card
from ..core.data_context import get_context
from ..core.theme import COLORS, page_header
from ..utils.ui import alert_component, filter_dropdown_col, kpi_card



def create_layout() -> html.Div:
    return dcc.Loading(
        id="loading-overview",
        type="cube",
        color=COLORS['primary'],
        children=_create_overview_content()
    )


def _create_overview_content() -> html.Div:
    ctx = get_context()
    info = ctx.eda.basic_info()
    diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'

    # Check model status
    from ..core.data_context import is_classifier_available
    model_available = is_classifier_available()
    if model_available:
        metrics = getattr(ctx.classifier, 'metrics', None)
        accuracy_text = f"{metrics.get('balanced_accuracy', 0)*100:.1f}%" if metrics else 'N/A'
        model_status = '✅ Treinado'
    else:
        accuracy_text = 'N/A'
        model_status = '⚠️ Não Treinado'

    # KPIs principais
    kpis_row = dbc.Row([
        kpi_card('📊', 'Total de Casos', f"{info['shape'][0]:,}", 'Registros no dataset', 'gradient_blue'),
        kpi_card('📈', 'Idade Média', f"{ctx.df['Idade'].mean():.1f} anos", f"Min: {ctx.df['Idade'].min()}, Max: {ctx.df['Idade'].max()}", 'gradient_primary'),
        kpi_card('👥', 'Distribuição', f"{ctx.df['Gênero'].value_counts().iloc[0]:,}", 'Maior grupo', 'gradient_success'),
        kpi_card('🏥', 'Diagnósticos', str(ctx.df[diagnosis_col].nunique()), 'Tipos únicos', 'gradient_warning'),
        kpi_card('🤖', 'Modelo ML', model_status, f'Acurácia: {accuracy_text}', 'gradient_teal'),
    ], style={'marginBottom': '30px'})

    # Header
    overview_header = page_header(
        'Visão Geral do Dataset',
        'Estatísticas essenciais, KPIs e distribuições principais',
        'Filtre por gênero e faixa etária para análises estratificadas',
    )

    # Filtros
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

    filters_section = dbc.Container([
        html.H3('🎯 Filtros de Estratificação', style={
            'color': COLORS['text'],
            'marginBottom': '10px',
            'fontSize': '1.6em',
            'fontWeight': '700'
        }),
        html.P('Customize a visualização por gênero e faixa etária', style={
            'color': COLORS['text_secondary'],
            'marginBottom': '20px',
            'fontSize': '0.95em'
        }),
        dbc.Row([
            filter_dropdown_col('overview-gender-filter', '👤 Gênero', gender_filter_options, 'todos'),
            filter_dropdown_col('overview-age-filter', '🎂 Faixa Etária', age_filter_options, 'todos'),
        ], style={
            'padding': '20px',
            'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
            'borderRadius': '14px',
            'border': f'1px solid {COLORS["border"]}'
        })
    ], fluid=True, style={'marginBottom': '30px'})

    # Alertas automáticos
    alerts_section = dbc.Container([
        html.Div(id='overview-alerts-container', style={'marginBottom': '20px'})
    ], fluid=True)

    # Seção de gráficos
    charts_section = dbc.Container([
        dbc.Row([
            dbc.Col([
                create_card([dcc.Graph(id='diagnosis-count-graph')], '📊 Distribuição de Diagnósticos')
            ], md=12, lg=8, style={'marginBottom': '20px'}),
            
            dbc.Col([
                create_card([dcc.Graph(id='gender-pie-chart')], '👥 Distribuição de Gênero')
            ], md=12, lg=4, style={'marginBottom': '20px'}),
        ]),

        dbc.Row([
            dbc.Col([
                create_card([dcc.Graph(id='age-dist-univariate')], '📊 Distribuição de Idade')
            ], md=12, lg=6, style={'marginBottom': '20px'}),
            
            dbc.Col([
                create_card([dcc.Graph(id='age-gender-heatmap')], '🔥 Heatmap: Idade vs Gênero')
            ], md=12, lg=6, style={'marginBottom': '20px'}),
        ]),

        dbc.Row([
            dbc.Col([
                create_card([dcc.Graph(id='climate-vars-distribution')], '🌡️ Distribuição de Variáveis Climáticas')
            ], md=12, style={'marginBottom': '20px'}),
        ]),

        dbc.Row([
            dbc.Col([
                create_card([dcc.Graph(id='diagnosis-age-violin')], '🎻 Violino: Diagnóstico por Idade')
            ], md=12, lg=6, style={'marginBottom': '20px'}),
            
            dbc.Col([
                create_card([dcc.Graph(id='top-diagnoses-by-gender')], '🏆 Top Diagnósticos por Gênero')
            ], md=12, lg=6, style={'marginBottom': '20px'}),
        ]),
    ], fluid=True)

    return dbc.Container([
        overview_header,
        kpis_row,
        alerts_section,
        filters_section,
        charts_section,
    ], fluid=True, style={
        'paddingTop': '20px',
        'paddingBottom': '40px'
    })


def register_callbacks(app) -> None:
    @app.callback(
        Output('overview-alerts-container', 'children'),
        [
            Input('tabs', 'value'),
            Input('overview-gender-filter', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_alerts(tab, gender, age_group):
        """Generate automatic alerts based on filtered data"""
        if tab != 'tab-overview':
            return []

        ctx = get_context()
        df_filtered = ctx.df.copy()
        
        # Aplicar filtros
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
        
        alerts = []
        
        # Alerta 1: Dados insuficientes
        if len(df_filtered) < 50:
            alerts.append(alert_component(
                'warning',
                'Dados Insuficientes',
                f'Apenas {len(df_filtered)} registros encontrados. Considere ajustar os filtros para análises mais robustas.'
            ))
        
        # Alerta 2: Classe desbalanceada
        diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
        if diagnosis_col in df_filtered.columns:
            diag_counts = df_filtered[diagnosis_col].value_counts()
            if len(diag_counts) > 0:
                max_prop = diag_counts.iloc[0] / len(df_filtered)
                if max_prop > 0.7:
                    alerts.append(alert_component(
                        'warning',
                        'Classe Desbalanceada',
                        f'Classe "{diag_counts.index[0]}" representa {max_prop*100:.1f}% dos dados. Modelos podem ser enviesados.'
                    ))
        
        # Alerta 3: Distribuição de gênero desigual
        gender_counts = df_filtered['Gênero'].value_counts()
        if len(gender_counts) == 2:
            gender_ratio = gender_counts.iloc[0] / gender_counts.iloc[1]
            if gender_ratio > 3 or gender_ratio < 0.33:
                alerts.append(alert_component(
                    'info',
                    'Gênero Desigualmente Distribuído',
                    f'Razão de gênero é {gender_ratio:.1f}:1. Possível enviesamento nos dados.'
                ))
        
        # Alerta 4: Dados completos
        if len(df_filtered) > 0 and len(alerts) == 0:
            alerts.append(alert_component(
                'success',
                'Dados Balanceados',
                f'{len(df_filtered)} registros com distribuição adequada para análise.'
            ))
        
        return alerts if alerts else []

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
            title='',
            labels={'Contagem': 'Número de Casos', 'Diagnóstico': 'Diagnóstico'}
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
            height=400
        )
        return fig

    @app.callback(
        Output('gender-pie-chart', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_gender_pie(tab, age_group):
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
        gender_counts['Gênero'] = gender_counts['Gênero'].map({0: '👩 Feminino', 1: '👨 Masculino'})

        fig = px.pie(
            gender_counts,
            values='Contagem',
            names='Gênero',
            color_discrete_map={'👩 Feminino': COLORS['accent'], '👨 Masculino': COLORS['primary']},
            title=''
        )

        fig.update_traces(
            textinfo='label+percent',
            textfont=dict(size=12, color=COLORS['text'], family='Inter, sans-serif'),
            hovertemplate='<b>%{label}</b><br>Casos: %{value}<br>Proporção: %{percent}<extra></extra>'
        )
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            font=dict(family='Inter, sans-serif', size=12),
            margin=dict(t=20, b=20, l=20, r=20),
            height=400
        )
        return fig

    @app.callback(
        Output('age-gender-heatmap', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-gender-filter', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_age_gender_heatmap(tab, gender, age_group):
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
        
        diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
        
        # Criar bins de idade
        age_bins = [0, 12, 17, 59, 120]
        age_labels = ['0-12', '13-17', '18-59', '60+']
        df_filtered['Faixa Etária'] = pd.cut(df_filtered['Idade'], bins=age_bins, labels=age_labels)
        
        # Criar matriz de cruzamento
        heatmap_data = pd.crosstab(df_filtered['Faixa Etária'], df_filtered[diagnosis_col])
        
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='Blues',
            labels={'x': 'Diagnóstico', 'y': 'Faixa Etária', 'color': 'Casos'},
            title=''
        )
        
        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            font=dict(family='Inter, sans-serif', size=11),
            height=350,
            xaxis_tickangle=-45
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
        
        fig = px.histogram(df_filtered, x='Idade', nbins=30, color_discrete_sequence=[COLORS['primary']], title='')

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
            height=400,
            margin=dict(t=30, b=60, l=60, r=30)
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
            font=dict(family='Inter, sans-serif', size=11)
        )
        return fig

    @app.callback(
        Output('diagnosis-age-violin', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-gender-filter', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_diagnosis_age_violin(tab, gender, age_group):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        df_filtered = ctx.df.copy()
        diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
        
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
        
        fig = px.violin(
            df_filtered,
            x=diagnosis_col,
            y='Idade',
            color=diagnosis_col,
            box=True,
            points=False,
            title='',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            xaxis_title='Diagnóstico',
            yaxis_title='Idade (anos)',
            showlegend=False,
            font=dict(family='Inter, sans-serif', size=11),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            xaxis_tickangle=-45,
            height=400,
            margin=dict(t=30, b=80, l=60, r=30)
        )
        return fig

    @app.callback(
        Output('top-diagnoses-by-gender', 'figure'),
        [
            Input('tabs', 'value'),
            Input('overview-age-filter', 'value'),
        ]
    )
    def update_top_diagnoses_gender(tab, age_group):
        if tab != 'tab-overview':
            return go.Figure()

        ctx = get_context()
        df_filtered = ctx.df.copy()
        diagnosis_col = ctx.diagnosis_cols[0] if ctx.diagnosis_cols else 'Diagnóstico'
        
        if age_group == 'crianca':
            df_filtered = df_filtered[df_filtered['Idade'] <= 12]
        elif age_group == 'adolescente':
            df_filtered = df_filtered[df_filtered['Idade'].between(13, 17)]
        elif age_group == 'adulto':
            df_filtered = df_filtered[df_filtered['Idade'].between(18, 59)]
        elif age_group == 'idoso':
            df_filtered = df_filtered[df_filtered['Idade'] >= 60]
        
        # Criar pivot table: top diagnósticos por gênero
        gender_map = {0: '👩 Feminino', 1: '👨 Masculino'}
        df_filtered['Gênero_Label'] = df_filtered['Gênero'].map(gender_map)
        
        top_diags = df_filtered[diagnosis_col].value_counts().head(8).index
        df_top = df_filtered[df_filtered[diagnosis_col].isin(top_diags)]
        
        counts = df_top.groupby(['Gênero_Label', diagnosis_col]).size().reset_index(name='count')
        
        fig = px.bar(
            counts,
            x=diagnosis_col,
            y='count',
            color='Gênero_Label',
            barmode='group',
            color_discrete_map={
                '👩 Feminino': COLORS['accent'],
                '👨 Masculino': COLORS['primary']
            },
            title='',
            labels={'count': 'Número de Casos', diagnosis_col: 'Diagnóstico'}
        )

        fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            showlegend=True,
            font=dict(family='Inter, sans-serif', size=11),
            xaxis=dict(gridcolor=COLORS['border'], showgrid=False),
            yaxis=dict(gridcolor=COLORS['border'], showgrid=True),
            xaxis_tickangle=-45,
            height=400,
            margin=dict(t=30, b=80, l=60, r=30),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        return fig


__all__ = ['create_layout', 'register_callbacks']

