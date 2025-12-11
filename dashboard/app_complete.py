# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Dashboard Principal - NimbusVita (Versão Completa)
Análise Exploratória de Doenças Relacionadas ao Clima
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from dashboard.core.theme import COLORS, INDEX_STRING
from dashboard.views import eda, overview, classification, pipeline_tuning, ml_clustering, prognosis
from dashboard.core.theme import _tab_style as theme_tab_style, _tab_selected_style as theme_tab_selected_style


# =====================================================================
# APP INITIALIZATION
# =====================================================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    suppress_callback_exceptions=True
)
app.title = "NimbusVita - Análise de Doenças Climáticas"
app.index_string = INDEX_STRING


# =====================================================================
# LAYOUT
# =====================================================================

def serve_layout():
    return html.Div(style={
        'backgroundColor': COLORS['background'], 
        'minHeight': '100vh', 
        'fontFamily': "'Inter', 'Segoe UI', 'Roboto', sans-serif",
        'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["background_light"]} 100%)'
    }, children=[
        # Store para tema
        dcc.Store(id='theme-store', data={'theme': 'dark'}),
        
        # Header
        html.Div(className='header', style={
            'position': 'relative',
            'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%)',
            'padding': '30px 20px',
            'marginBottom': '30px',
            'boxShadow': '0 10px 30px rgba(0,0,0,0.5)',
            'borderBottom': f'3px solid {COLORS["accent"]}'
        }, children=[
            html.Div(style={'maxWidth': '1400px', 'margin': '0 auto'}, children=[
                html.H1('NimbusVita', 
                        style={
                            'color': 'white', 
                            'textAlign': 'center', 
                            'margin': '0', 
                            'fontSize': '3em',
                            'fontWeight': '700',
                            'letterSpacing': '1px',
                            'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
                        }),
                html.P('Weather Related Disease Analysis',
                       style={
                           'color': 'rgba(255,255,255,0.95)', 
                           'textAlign': 'center', 
                           'margin': '10px 0 5px 0', 
                           'fontSize': '1.3em',
                           'fontWeight': '500'
                       }),
                html.P('Análise Exploratória de Doenças Relacionadas ao Clima',
                       style={
                           'color': 'rgba(255,255,255,0.85)', 
                           'textAlign': 'center', 
                           'margin': '5px 0 0 0', 
                           'fontSize': '0.95em',
                           'fontWeight': '400'
                       })
            ])
        ]),
        
        # Container para tabs
        html.Div(style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '0 20px'}, children=[
            # Tabs de navegação
            dcc.Tabs(id='tabs', value='tab-overview', style={
                'backgroundColor': 'transparent',
                'borderBottom': f'2px solid {COLORS["border"]}'
            }, children=[
                dcc.Tab(label='Visão Geral', value='tab-overview', 
                        style=_tab_style(),
                        selected_style=_tab_selected_style()),
                dcc.Tab(label='Análise Exploratória', value='tab-eda', 
                        style=_tab_style(),
                        selected_style=_tab_selected_style()),
                dcc.Tab(label='Classificação & SHAP', value='tab-classification', 
                        style=_tab_style(),
                        selected_style=_tab_selected_style()),
                dcc.Tab(label='Clusterização', value='tab-clustering', 
                        style=_tab_style(),
                        selected_style=_tab_selected_style()),
                dcc.Tab(label='Prognóstico', value='tab-prognosis', 
                        style=_tab_style(),
                        selected_style=_tab_selected_style()),
                dcc.Tab(label='Pipeline & Tuning', value='tab-pipeline-tuning', 
                        style=_tab_style(),
                        selected_style=_tab_selected_style()),
            ]),
            
            # Conteúdo das tabs
            html.Div(id='tabs-content', style={'padding': '30px 0'})
        ])
    ])
    # Delegate tab style helpers to the theme module to keep layout clean

def _tab_style():
    return theme_tab_style()

def _tab_selected_style():
    return theme_tab_selected_style()


# Assign layout after helper functions exist
app.layout = serve_layout()  

# =====================================================================
# CALLBACKS
# =====================================================================
app.clientside_callback(
    """
    function(data) {
        if (data && data.theme) {
            localStorage.setItem('nimbusvita-theme', data.theme);
            sessionStorage.setItem('nimbusvita-current-theme', data.theme);
        }
        return data;
    }
    """,
    Output('theme-store', 'id'),
    Input('theme-store', 'data'),
    prevent_initial_call=True
)


@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    """Route to appropriate view based on selected tab."""
    if tab == 'tab-overview':
        return overview.create_layout()
    elif tab == 'tab-eda':
        return eda.create_layout()
    elif tab == 'tab-classification':
        return classification.create_layout()
    elif tab == 'tab-clustering':
        return ml_clustering.create_layout()
    elif tab == 'tab-prognosis':
        return prognosis.create_prognosis_tab()
    elif tab == 'tab-pipeline-tuning':
        return pipeline_tuning.create_layout()
    return html.Div('Tab não implementada')


# =====================================================================
# REGISTER VIEW CALLBACKS
# =====================================================================

overview.register_callbacks(app)
eda.register_callbacks(app)
classification.register_callbacks(app)
ml_clustering.register_callbacks(app)
pipeline_tuning.register_callbacks(app)


# =====================================================================
# RUN APP
# =====================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8050)
