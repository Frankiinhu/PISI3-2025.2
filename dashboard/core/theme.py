"""UI theme helpers for the NimbusVita dashboard."""
from __future__ import annotations

from typing import Dict

from dash import html
import plotly.graph_objects as go

COLORS: Dict[str, str] = {
    'background': '#0a0e27',
    'background_light': '#1a1f3a',
    'primary': '#5559ff',
    'primary_light': '#7b7fff',
    'primary_dark': '#3d41cc',
    'secondary': '#131829',
    'secondary_light': '#f5d76e',
    'secondary_dark': '#d1974b',
    'text': '#e8eaf6',
    'text_secondary': '#9fa8da',
    'accent': '#a4a8ff',
    'accent_light': '#c4c7ff',
    'accent_secondary': '#4facfe',
    'card': '#1e2139',
    'card_hover': '#252a48',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'error': '#f87171',
    'border': '#2d3250',
}

INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            /* Animações suaves */
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 48px rgba(0,0,0,0.6) !important;
            }
            
            .card-hover:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 48px rgba(102, 126, 234, 0.3) !important;
            }
            
            /* Estilo para inputs */
            input[type="number"] {
                background-color: rgba(255,255,255,0.05) !important;
                border: 1px solid #2d3250 !important;
                color: #e8eaf6 !important;
                transition: all 0.3s ease;
            }
            
            input[type="number"]:focus {
                border-color: #5559ff !important;
                box-shadow: 0 0 0 3px rgba(85, 89, 255, 0.1) !important;
                outline: none;
            }
            
            /* Estilo para checkboxes */
            input[type="checkbox"] {
                accent-color: #5559ff;
            }
            
            /* Animação do botão */
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 40px rgba(85, 89, 255, 0.6) !important;
            }
            
            button:active {
                transform: translateY(0);
            }
            
            /* Scrollbar customizada */
            ::-webkit-scrollbar {
                width: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #0a0e27;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #5559ff 0%, #7b7fff 100%);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #7b7fff 0%, #a4a8ff 100%);
            }
            
            /* Dropdown styles */
            .Select-control {
                background-color: rgba(255,255,255,0.05) !important;
                border-color: #2d3250 !important;
            }
            
            .Select-menu-outer {
                background-color: #1e2139 !important;
                border: 1px solid #2d3250 !important;
            }
            
            .Select-option {
                background-color: #1e2139 !important;
                color: #e8eaf6 !important;
            }
            
            .Select-option:hover {
                background-color: #252a48 !important;
            }
            
            .custom-dropdown .Select-value-label,
            .custom-dropdown .Select-placeholder,
            .custom-dropdown input {
                color: #e8eaf6 !important;
            }
            
            .custom-dropdown .Select-value {
                background-color: rgba(102, 126, 234, 0.2) !important;
                border-color: rgba(102, 126, 234, 0.4) !important;
                color: #e8eaf6 !important;
            }
            
            .custom-dropdown .Select-input input {
                color: #e8eaf6 !important;
            }
            
            div[class*="css-"] input {
                color: #e8eaf6 !important;
            }
            
            div[class*="singleValue"] {
                color: #e8eaf6 !important;
            }
            
            div[class*="placeholder"] {
                color: rgba(232, 234, 246, 0.6) !important;
            }
            
            ._dash-undo-redo {
                display: none;
            }
            
            .tab {
                transition: all 0.3s ease;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            
            .js-plotly-plot {
                animation: fadeInUp 0.8s ease-out;
            }
            
            ._dash-loading {
                position: relative;
            }
            
            ._dash-loading::after {
                content: "";
                position: absolute;
                top: 50%;
                left: 50%;
                width: 50px;
                height: 50px;
                margin: -25px 0 0 -25px;
                border: 4px solid rgba(85, 89, 255, 0.3);
                border-top-color: #5559ff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            [id*="card-"] {
                animation: fadeInUp 0.6s ease-out;
                animation-fill-mode: both;
            }
            
            [id*="card-"]:hover {
                transform: translateY(-5px) scale(1.02);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            @keyframes skeletonLoading {
                0% { background-position: -200px 0; }
                100% { background-position: calc(200px + 100%) 0; }
            }
            
            .skeleton-loader {
                background: linear-gradient(
                    90deg,
                    rgba(85, 89, 255, 0.1) 0%,
                    rgba(85, 89, 255, 0.3) 50%,
                    rgba(85, 89, 255, 0.1) 100%
                );
                background-size: 200px 100%;
                animation: skeletonLoading 1.5s infinite;
                border-radius: 8px;
            }
            
            .loading-state {
                opacity: 0.6;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }
            
            @keyframes successPulse {
                0%, 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
                50% { box-shadow: 0 0 0 20px rgba(74, 222, 128, 0); }
            }
            
            .success-feedback { animation: successPulse 1s ease-out; }
            
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            .bounce-animation { animation: bounce 2s infinite; }
            
            @keyframes slideInLeft {
                from { opacity: 0; transform: translateX(-50px); }
                to { opacity: 1; transform: translateX(0); }
            }
            
            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(50px); }
                to { opacity: 1; transform: translateX(0); }
            }
            
            .slide-in-left { animation: slideInLeft 0.6s ease-out; }
            .slide-in-right { animation: slideInRight 0.6s ease-out; }
            
            @keyframes progressBar {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            
            .progress-bar {
                height: 4px;
                background: linear-gradient(90deg, #5559ff, #7b7fff, #a4a8ff);
                animation: progressBar 2s ease-out;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def metrics_unavailable_figure(message: str = 'Execute o treinamento e salve o modelo para visualizar as métricas registradas.') -> go.Figure:
    """Return a placeholder figure when model metrics are missing."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS['text'], family='Inter, sans-serif'),
    )
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], family='Inter, sans-serif'),
        margin=dict(t=40, b=40, l=40, r=40),
        height=400,
    )
    return fig


def page_header(title: str, subtitle: str, description: str) -> html.Div:
    """Utility component shared across layouts."""
    return html.Div([
        html.H2(title, style={'color': COLORS['text'], 'marginBottom': '10px', 'fontSize': '2em', 'fontWeight': '700'}),
        html.P(subtitle, style={'color': COLORS['text_secondary'], 'fontSize': '1em', 'marginBottom': '30px'}),
        html.P(description, style={'color': COLORS['text_secondary'], 'fontSize': '0.95em', 'marginBottom': '30px'}) if description else None,
    ])


__all__ = ['COLORS', 'INDEX_STRING', 'metrics_unavailable_figure', 'page_header']
