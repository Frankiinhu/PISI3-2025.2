"""
Dashboard Principal - VitaNimbus (Versão Completa com Callbacks)
Análise Exploratória de Dados e Predição de Doenças Relacionadas ao Clima
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

# Adicionar diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import DataLoader
from src.data_processing.eda import ExploratoryDataAnalysis
from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer

# Variáveis globais que serão carregadas sob demanda
df_global = None
eda_global = None
classifier = None
clusterer = None
loader = None
symptom_cols = []
climatic_vars = []

def load_data_and_models():
    """Carrega dados e modelos apenas uma vez"""
    global df_global, eda_global, classifier, clusterer, loader, symptom_cols, climatic_vars
    
    if df_global is not None:
        return  # Já carregado
    
    print("Carregando dados...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DATASET FINAL WRDP.csv')
    loader = DataLoader(data_path)
    df_global = loader.get_clean_data()
    eda_global = ExploratoryDataAnalysis(df_global)
    
    # Obter feature names
    feature_dict = loader.get_feature_names()
    symptom_cols = feature_dict['symptoms']
    climatic_vars = feature_dict['climatic']
    
    # Carregar modelos
    print("Carregando modelos...")
    classifier = DiagnosisClassifier()
    clusterer = DiseaseClusterer()
    
    try:
        classifier_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'classifier_model.pkl')
        classifier.load_model(classifier_path)
        print("✓ Classificador carregado")
    except Exception as e:
        print(f"⚠ Classificador não carregado: {e}")
    
    try:
        clusterer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'clustering_model.pkl')
        clusterer.load_model(clusterer_path)
        print("✓ Clusterizador carregado")
    except Exception as e:
        print(f"⚠ Clusterizador não carregado: {e}")

# Inicializar app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "VitaNimbus - Análise de Doenças Climáticas"

# Cores e estilo
COLORS = {
    'background': '#0f1419',
    'primary': '#1DA1F2',
    'secondary': '#14171A',
    'text': '#E1E8ED',
    'accent': '#00C9A7',
    'card': '#192734'
}

# Layout principal
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[
    # Header
    html.Div(className='header', style={
        'backgroundColor': COLORS['secondary'],
        'padding': '20px',
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.3)'
    }, children=[
        html.H1('🌡️ VitaNimbus - Weather Related Disease Prediction', 
                style={'color': COLORS['primary'], 'textAlign': 'center', 'margin': '0', 'fontSize': '2.5em'}),
        html.P('Análise Exploratória de Dados e Predição de Doenças Relacionadas ao Clima',
               style={'color': COLORS['text'], 'textAlign': 'center', 'margin': '10px 0 0 0', 'fontSize': '1.1em'})
    ]),
    
    # Tabs de navegação
    dcc.Tabs(id='tabs', value='tab-overview', style={
        'backgroundColor': COLORS['secondary']
    }, children=[
        dcc.Tab(label='📊 Visão Geral', value='tab-overview', 
                style={'color': COLORS['text'], 'backgroundColor': COLORS['secondary']},
                selected_style={'color': COLORS['primary'], 'backgroundColor': COLORS['card'], 'fontWeight': 'bold'}),
        dcc.Tab(label='🔍 Análise Exploratória', value='tab-eda', 
                style={'color': COLORS['text'], 'backgroundColor': COLORS['secondary']},
                selected_style={'color': COLORS['primary'], 'backgroundColor': COLORS['card'], 'fontWeight': 'bold'}),
        dcc.Tab(label='🌡️ Clima vs Diagnóstico', value='tab-climate', 
                style={'color': COLORS['text'], 'backgroundColor': COLORS['secondary']},
                selected_style={'color': COLORS['primary'], 'backgroundColor': COLORS['card'], 'fontWeight': 'bold'}),
        dcc.Tab(label='💊 Sintomas', value='tab-symptoms', 
                style={'color': COLORS['text'], 'backgroundColor': COLORS['secondary']},
                selected_style={'color': COLORS['primary'], 'backgroundColor': COLORS['card'], 'fontWeight': 'bold'}),
        dcc.Tab(label='🤖 Modelos ML', value='tab-ml', 
                style={'color': COLORS['text'], 'backgroundColor': COLORS['secondary']},
                selected_style={'color': COLORS['primary'], 'backgroundColor': COLORS['card'], 'fontWeight': 'bold'}),
        dcc.Tab(label='🎯 Predição', value='tab-prediction', 
                style={'color': COLORS['text'], 'backgroundColor': COLORS['secondary']},
                selected_style={'color': COLORS['primary'], 'backgroundColor': COLORS['card'], 'fontWeight': 'bold'}),
    ]),
    
    # Conteúdo das tabs
    html.Div(id='tabs-content', style={'padding': '20px'})
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    """Renderiza conteúdo baseado na tab selecionada"""
    load_data_and_models()  # Garante que dados estão carregados
    
    if tab == 'tab-overview':
        return create_overview_layout()
    elif tab == 'tab-eda':
        return create_eda_layout()
    elif tab == 'tab-climate':
        return create_climate_layout()
    elif tab == 'tab-symptoms':
        return create_symptoms_layout()
    elif tab == 'tab-ml':
        return create_ml_layout()
    elif tab == 'tab-prediction':
        return create_prediction_layout()


def create_card(children, title=None):
    """Cria um card estilizado"""
    content = []
    if title:
        content.append(html.H3(title, style={'color': COLORS['primary'], 'marginBottom': '15px'}))
    content.extend(children if isinstance(children, list) else [children])
    
    return html.Div(
        content,
        style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '20px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
        }
    )


def create_overview_layout():
    """Layout da visão geral"""
    info = eda_global.basic_info()
    
    # Estatísticas básicas
    stats_cards = html.Div([
        html.Div([
            html.Div([
                html.H4('📋 Total de Registros', style={'color': COLORS['text'], 'margin': '0'}),
                html.H2(f"{info['shape'][0]:,}", style={'color': COLORS['accent'], 'margin': '10px 0'}),
            ], style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px', 
                     'textAlign': 'center', 'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Div([
                html.H4('🏥 Diagnósticos Únicos', style={'color': COLORS['text'], 'margin': '0'}),
                html.H2(f"{df_global['Diagnóstico'].nunique()}", style={'color': COLORS['accent'], 'margin': '10px 0'}),
            ], style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px',
                     'textAlign': 'center', 'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Div([
                html.H4('📊 Total de Features', style={'color': COLORS['text'], 'margin': '0'}),
                html.H2(f"{info['shape'][1]}", style={'color': COLORS['accent'], 'margin': '10px 0'}),
            ], style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px',
                     'textAlign': 'center', 'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Div([
                html.H4('🔬 Sintomas Analisados', style={'color': COLORS['text'], 'margin': '0'}),
                html.H2(f"{len(symptom_cols)}", style={'color': COLORS['accent'], 'margin': '10px 0'}),
            ], style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px',
                     'textAlign': 'center', 'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),
    ])
    
    return html.Div([
        html.H2('📊 Visão Geral do Dataset', style={'color': COLORS['primary'], 'marginBottom': '20px'}),
        
        stats_cards,
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='diagnosis-count-graph')], 'Distribuição de Diagnósticos')
            ], style={'width': '100%', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='age-distribution-graph')], 'Distribuição de Idade')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                create_card([dcc.Graph(id='climate-vars-distribution')], 'Variáveis Climáticas')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
    ])


@app.callback(
    Output('diagnosis-count-graph', 'figure'),
    Input('tabs', 'value')
)
def update_diagnosis_count(tab):
    """Atualiza gráfico de contagem de diagnósticos"""
    load_data_and_models()
    if tab != 'tab-overview':
        return {}
    
    diag_counts = df_global['Diagnóstico'].value_counts().reset_index()
    diag_counts.columns = ['Diagnóstico', 'Contagem']
    
    fig = px.bar(diag_counts, x='Diagnóstico', y='Contagem',
                 title='',
                 color='Contagem',
                 color_continuous_scale='Blues')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Diagnóstico',
        yaxis_title='Número de Casos',
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('age-distribution-graph', 'figure'),
    Input('tabs', 'value')
)
def update_age_distribution(tab):
    """Atualiza distribuição de idade"""
    load_data_and_models()
    if tab != 'tab-overview':
        return {}
    
    fig = px.histogram(df_global, x='Idade', nbins=30,
                      title='',
                      color_discrete_sequence=[COLORS['accent']])
    
    mean_age = df_global['Idade'].mean()
    fig.add_vline(x=mean_age, line_dash="dash", line_color="red",
                  annotation_text=f"Média: {mean_age:.1f} anos")
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Idade (anos)',
        yaxis_title='Frequência',
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('climate-vars-distribution', 'figure'),
    Input('tabs', 'value')
)
def update_climate_distribution(tab):
    """Atualiza distribuição de variáveis climáticas"""
    load_data_and_models()
    if tab != 'tab-overview':
        return {}
    
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=climatic_vars)
    
    colors = [COLORS['primary'], COLORS['accent'], '#FF6B6B']
    
    for i, var in enumerate(climatic_vars, 1):
        fig.add_trace(
            go.Histogram(x=df_global[var], name=var, marker_color=colors[i-1]),
            row=i, col=1
        )
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False
    )
    
    return fig


def create_eda_layout():
    """Layout da análise exploratória"""
    return html.Div([
        html.H2('🔍 Análise Exploratória de Dados', style={'color': COLORS['primary'], 'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Selecione Sintomas para Análise:', 
                      style={'color': COLORS['text'], 'fontSize': '1.1em', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='symptom-selector',
                options=[{'label': s, 'value': s} for s in symptom_cols[:20]],  # Top 20 sintomas
                value=symptom_cols[:4] if len(symptom_cols) >= 4 else symptom_cols,
                multi=True,
                placeholder='Selecione sintomas...',
                style={'backgroundColor': COLORS['secondary'], 'color': COLORS['text']}
            )
        ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': COLORS['card'], 
                 'borderRadius': '10px'}),
        
        html.Div([
            create_card([dcc.Graph(id='symptom-frequency-graphs')], 
                       'Frequência de Sintomas por Diagnóstico')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='correlation-matrix-graph')], 
                       'Matriz de Correlação (Top Features)')
        ]),
    ])


@app.callback(
    Output('symptom-frequency-graphs', 'figure'),
    [Input('symptom-selector', 'value'),
     Input('tabs', 'value')]
)
def update_symptom_frequency(selected_symptoms, tab):
    """Atualiza gráficos de frequência de sintomas"""
    load_data_and_models()
    if tab != 'tab-eda' or not selected_symptoms:
        return {}
    
    # Criar subplots
    n_symptoms = len(selected_symptoms)
    rows = (n_symptoms + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=selected_symptoms,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, symptom in enumerate(selected_symptoms):
        if symptom in df_global.columns:
            freq = df_global.groupby('Diagnóstico')[symptom].sum().reset_index()
            freq.columns = ['Diagnóstico', 'Contagem']
            
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fig.add_trace(
                go.Bar(x=freq['Diagnóstico'], y=freq['Contagem'], 
                      name=symptom, showlegend=False,
                      marker_color=COLORS['accent']),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * rows,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text']
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


@app.callback(
    Output('correlation-matrix-graph', 'figure'),
    Input('tabs', 'value')
)
def update_correlation_matrix(tab):
    """Atualiza matriz de correlação"""
    load_data_and_models()
    if tab != 'tab-eda':
        return {}
    
    # Selecionar top features
    if classifier.feature_importances is not None:
        top_features = classifier.feature_importances.head(15).index.tolist()
        # Adicionar variáveis climáticas
        features_to_correlate = list(set(top_features + climatic_vars + ['Idade']))
        features_to_correlate = [f for f in features_to_correlate if f in df_global.columns]
    else:
        features_to_correlate = climatic_vars + ['Idade'] + symptom_cols[:10]
    
    corr_matrix = df_global[features_to_correlate].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title='Correlação')
    ))
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_climate_layout():
    """Layout de análise climática"""
    return html.Div([
        html.H2('🌡️ Relação entre Clima e Diagnósticos', 
               style={'color': COLORS['primary'], 'marginBottom': '20px'}),
        
        html.Div([
            create_card([dcc.Graph(id='temp-diagnosis-graph')], 
                       'Temperatura vs Diagnóstico')
        ]),
        
        html.Div([
            html.Div([
                create_card([dcc.Graph(id='humidity-diagnosis-graph')], 
                           'Umidade vs Diagnóstico')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                create_card([dcc.Graph(id='wind-diagnosis-graph')], 
                           'Velocidade do Vento vs Diagnóstico')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
    ])


@app.callback(
    Output('temp-diagnosis-graph', 'figure'),
    Input('tabs', 'value')
)
def update_temp_diagnosis(tab):
    """Atualiza gráfico temperatura vs diagnóstico"""
    load_data_and_models()
    if tab != 'tab-climate':
        return {}
    
    fig = px.box(df_global, x='Diagnóstico', y='Temperatura (°C)',
                 color='Diagnóstico',
                 title='')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False,
        xaxis_tickangle=45
    )
    
    return fig


@app.callback(
    Output('humidity-diagnosis-graph', 'figure'),
    Input('tabs', 'value')
)
def update_humidity_diagnosis(tab):
    """Atualiza gráfico umidade vs diagnóstico"""
    load_data_and_models()
    if tab != 'tab-climate':
        return {}
    
    fig = px.box(df_global, x='Diagnóstico', y='Umidade',
                 color='Diagnóstico',
                 title='')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False,
        xaxis_tickangle=45
    )
    
    return fig


@app.callback(
    Output('wind-diagnosis-graph', 'figure'),
    Input('tabs', 'value')
)
def update_wind_diagnosis(tab):
    """Atualiza gráfico vento vs diagnóstico"""
    load_data_and_models()
    if tab != 'tab-climate':
        return {}
    
    fig = px.box(df_global, x='Diagnóstico', y='Velocidade do Vento (km/h)',
                 color='Diagnóstico',
                 title='')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        showlegend=False,
        xaxis_tickangle=45
    )
    
    return fig


def create_symptoms_layout():
    """Layout de análise de sintomas"""
    return html.Div([
        html.H2('💊 Análise de Sintomas', style={'color': COLORS['primary'], 'marginBottom': '20px'}),
        
        html.Div([
            create_card([dcc.Graph(id='symptom-heatmap-graph')], 
                       'Heatmap de Sintomas por Diagnóstico (Top 15)')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='symptom-importance-graph')], 
                       'Top 15 Sintomas por Importância')
        ]),
    ])


@app.callback(
    Output('symptom-heatmap-graph', 'figure'),
    Input('tabs', 'value')
)
def update_symptom_heatmap(tab):
    """Atualiza heatmap de sintomas"""
    load_data_and_models()
    if tab != 'tab-symptoms':
        return {}
    
    # Agrupar sintomas por diagnóstico
    symptom_by_diagnosis = df_global.groupby('Diagnóstico')[symptom_cols].sum()
    
    # Selecionar top 15 sintomas mais frequentes
    top_symptoms = symptom_by_diagnosis.sum().sort_values(ascending=False).head(15).index
    symptom_subset = symptom_by_diagnosis[top_symptoms]
    
    fig = go.Figure(data=go.Heatmap(
        z=symptom_subset.T.values,
        x=symptom_subset.index,
        y=top_symptoms,
        colorscale='Viridis',
        colorbar=dict(title='Contagem')
    ))
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_tickangle=45
    )
    
    return fig


@app.callback(
    Output('symptom-importance-graph', 'figure'),
    Input('tabs', 'value')
)
def update_symptom_importance(tab):
    """Atualiza gráfico de importância de sintomas"""
    load_data_and_models()
    if tab != 'tab-symptoms':
        return {}
    
    if classifier.feature_importances is None:
        return {}
    
    top_features = classifier.feature_importances.head(15)
    
    fig = px.bar(x=top_features.values, y=top_features.index,
                 orientation='h',
                 title='',
                 color=top_features.values,
                 color_continuous_scale='Blues')
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Importância',
        yaxis_title='Feature',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_ml_layout():
    """Layout dos modelos de ML"""
    if classifier.model is None:
        return html.Div([
            html.H2('🤖 Modelos de Machine Learning', style={'color': COLORS['primary']}),
            html.P('⚠️ Modelos não carregados. Execute o script de treinamento primeiro.',
                  style={'color': COLORS['text'], 'fontSize': '1.2em'})
        ])
    
    return html.Div([
        html.H2('🤖 Modelos de Machine Learning', style={'color': COLORS['primary'], 'marginBottom': '20px'}),
        
        html.Div([
            create_card([dcc.Graph(id='feature-importance-graph')], 
                       'Top 20 Features Mais Importantes (Random Forest)')
        ]),
        
        html.Div([
            create_card([dcc.Graph(id='cluster-visualization-graph')], 
                       'Visualização de Clusters (PCA 2D)')
        ]),
    ])


@app.callback(
    Output('feature-importance-graph', 'figure'),
    Input('tabs', 'value')
)
def update_feature_importance(tab):
    """Atualiza gráfico de importância de features"""
    load_data_and_models()
    if tab != 'tab-ml' or classifier.feature_importances is None:
        return {}
    
    top_20 = classifier.feature_importances.head(20)
    
    fig = px.bar(x=top_20.values, y=top_20.index,
                 orientation='h',
                 title='',
                 color=top_20.values,
                 color_continuous_scale='Bluered')
    
    fig.update_layout(
        height=600,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title='Importância',
        yaxis_title='Feature',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


@app.callback(
    Output('cluster-visualization-graph', 'figure'),
    Input('tabs', 'value')
)
def update_cluster_visualization(tab):
    """Atualiza visualização de clusters"""
    load_data_and_models()
    if tab != 'tab-ml' or clusterer.labels_ is None:
        return {}
    
    # Preparar dados para visualização
    X_scaled = clusterer.scaler.transform(
        df_global.select_dtypes(include=[np.number]).drop('Diagnóstico', axis=1, errors='ignore')
    )
    
    # PCA para 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Criar DataFrame para plotar
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusterer.labels_.astype(str),
        'Diagnóstico': df_global['Diagnóstico'].values
    })
    
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                    hover_data=['Diagnóstico'],
                    title='',
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font_color=COLORS['text'],
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)',
        height=600
    )
    
    return fig


def create_prediction_layout():
    """Layout de predição"""
    return html.Div([
        html.H2('🎯 Sistema de Predição de Diagnóstico', 
               style={'color': COLORS['primary'], 'marginBottom': '20px'}),
        
        create_card([
            html.P('Insira os dados do paciente para obter uma predição de diagnóstico:',
                  style={'color': COLORS['text'], 'fontSize': '1.1em', 'marginBottom': '20px'}),
            
            # Inputs climáticos e demográficos
            html.Div([
                html.Div([
                    html.Label('Idade:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-age', type='number', value=35,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Temperatura (°C):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-temp', type='number', value=25.0, step=0.1,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Umidade (0-1):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-humidity', type='number', value=0.7, step=0.01, min=0, max=1,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
                
                html.Div([
                    html.Label('Velocidade do Vento (km/h):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(id='input-wind', type='number', value=10.0, step=0.1,
                             style={'width': '100%', 'padding': '10px', 'borderRadius': '5px',
                                   'border': 'none', 'fontSize': '1em'})
                ], style={'width': '23%', 'display': 'inline-block', 'margin': '10px'}),
            ]),
            
            # Seleção de sintomas
            html.Div([
                html.Label('Selecione Sintomas Presentes:', 
                          style={'color': COLORS['text'], 'fontSize': '1.1em', 
                                'fontWeight': 'bold', 'marginTop': '20px'}),
                dcc.Checklist(
                    id='symptom-checklist',
                    options=[{'label': f' {s}', 'value': s} for s in symptom_cols[:30]],
                    value=[],
                    style={'color': COLORS['text'], 'columnCount': 3, 'marginTop': '15px'},
                    labelStyle={'display': 'block', 'marginBottom': '10px'}
                )
            ], style={'marginTop': '20px'}),
            
            html.Button('🔍 Fazer Predição', id='predict-button', 
                       style={
                           'backgroundColor': COLORS['primary'],
                           'color': 'white',
                           'padding': '15px 40px',
                           'border': 'none',
                           'borderRadius': '5px',
                           'fontSize': '1.1em',
                           'cursor': 'pointer',
                           'marginTop': '30px',
                           'fontWeight': 'bold',
                           'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'
                       }),
            
            html.Div(id='prediction-result', style={'marginTop': '30px'})
        ])
    ])


@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('input-age', 'value'),
     State('input-temp', 'value'),
     State('input-humidity', 'value'),
     State('input-wind', 'value'),
     State('symptom-checklist', 'value')]
)
def make_prediction(n_clicks, age, temp, humidity, wind, symptoms):
    """Faz predição de diagnóstico"""
    load_data_and_models()
    if n_clicks is None or classifier.model is None:
        return html.Div()
    
    try:
        # Construir features
        features = {
            'Idade': age or 35,
            'Temperatura (°C)': temp or 25.0,
            'Umidade': humidity or 0.7,
            'Velocidade do Vento (km/h)': wind or 10.0
        }
        
        # Adicionar sintomas
        for symptom in classifier.feature_names:
            if symptom not in features:
                features[symptom] = 1 if symptom in (symptoms or []) else 0
        
        # Criar DataFrame
        X = pd.DataFrame([features])
        X = X[classifier.feature_names]
        
        # Fazer predição
        diagnosis = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]
        confidence = max(probabilities) * 100
        
        # Preparar resultado
        all_probs = sorted(zip(classifier.label_encoder.classes_, probabilities), 
                          key=lambda x: x[1], reverse=True)
        
        result = html.Div([
            html.Div([
                html.H3('🎯 Resultado da Predição', 
                       style={'color': COLORS['primary'], 'marginBottom': '20px'}),
                
                html.Div([
                    html.H4('Diagnóstico Predito:', style={'color': COLORS['text']}),
                    html.H2(diagnosis, style={'color': COLORS['accent'], 'margin': '10px 0'}),
                    html.P(f'Confiança: {confidence:.2f}%', 
                          style={'color': COLORS['text'], 'fontSize': '1.2em'})
                ], style={'backgroundColor': COLORS['background'], 'padding': '20px', 
                         'borderRadius': '10px', 'marginBottom': '20px', 'textAlign': 'center'}),
                
                html.H4('📊 Probabilidades por Diagnóstico:', 
                       style={'color': COLORS['text'], 'marginTop': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Span(f'{diag}: ', style={'fontWeight': 'bold', 'color': COLORS['text']}),
                        html.Span(f'{prob*100:.2f}%', 
                                style={'color': COLORS['accent'] if prob == max(probabilities) else COLORS['text']})
                    ], style={'marginBottom': '10px'})
                    for diag, prob in all_probs[:5]
                ])
            ], style={'backgroundColor': COLORS['card'], 'padding': '30px', 
                     'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.3)'})
        ])
        
        return result
        
    except Exception as e:
        return html.Div([
            html.P(f'❌ Erro na predição: {str(e)}', 
                  style={'color': 'red', 'fontSize': '1.1em'})
        ])


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌡️ VitaNimbus Dashboard")
    print("="*60)
    print("✓ Dashboard iniciado com sucesso!")
    print("📊 Acesse: http://127.0.0.1:8050/")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)
