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
    # Gradientes
    'gradient_primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_success': 'linear-gradient(135deg, #4ade80 0%, #22c55e 100%)',
    'gradient_warning': 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
    'gradient_error': 'linear-gradient(135deg, #f87171 0%, #ef4444 100%)',
    'gradient_accent': 'linear-gradient(135deg, #667eea 0%, #4facfe 100%)',
    'gradient_purple': 'linear-gradient(135deg, #a855f7 0%, #8b5cf6 100%)',
    'gradient_blue': 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
    'gradient_teal': 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
    'gradient_card': 'linear-gradient(135deg, #1e2139 0%, #252a48 100%)',
    # Glassmorphism
    'glass_light': 'rgba(255, 255, 255, 0.05)',
    'glass_medium': 'rgba(255, 255, 255, 0.1)',
    'glass_border': 'rgba(255, 255, 255, 0.18)',
    # Estados e feedback
    'info': '#60a5fa',
    'info_light': '#93c5fd',
    'hover_overlay': 'rgba(102, 126, 234, 0.1)',
    'active_overlay': 'rgba(102, 126, 234, 0.2)',
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
                background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1836 100%);
                min-height: 100vh;
            }
            
            /* Animações suaves e modernas */
            .stat-card:hover {
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 20px 60px rgba(102, 126, 234, 0.35) !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .stat-card {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                background: linear-gradient(135deg, rgba(30, 33, 57, 0.8), rgba(37, 42, 72, 0.8));
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }
            
            .card-hover:hover {
                transform: translateY(-6px);
                box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3) !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .card-hover {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Estilo para inputs com glassmorphism */
            input[type="number"],
            input[type="text"],
            input[type="date"],
            select {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                color: #e8eaf6 !important;
                transition: all 0.3s ease;
                border-radius: 8px;
                padding: 10px 12px;
                backdrop-filter: blur(10px);
            }
            
            input[type="number"]:hover,
            input[type="text"]:hover,
            input[type="date"]:hover,
            select:hover {
                border-color: rgba(102, 126, 234, 0.3) !important;
                background-color: rgba(255, 255, 255, 0.08) !important;
            }
            
            input[type="number"]:focus,
            input[type="text"]:focus,
            input[type="date"]:focus,
            select:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
                outline: none;
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            
            /* Estilo para checkboxes */
            input[type="checkbox"] {
                accent-color: #667eea;
                transition: transform 0.2s ease;
                cursor: pointer;
            }
            
            input[type="checkbox"]:hover {
                transform: scale(1.1);
            }
            
            /* Animação do botão melhorada */
            button {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            button:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
            }
            
            button:active {
                transform: translateY(-1px);
            }
            
            button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                transition: left 0.5s ease;
            }
            
            button:hover::before {
                left: 100%;
            }
            
            /* Scrollbar customizada com gradiente */
            ::-webkit-scrollbar {
                width: 12px;
                height: 12px;
            }
            
            ::-webkit-scrollbar-track {
                background: transparent;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #5559ff 0%, #667eea 50%, #764ba2 100%);
                border-radius: 10px;
                transition: all 0.3s ease;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #a855f7 100%);
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.4);
            }
            
            /* Dropdown styles com glassmorphism */
            .Select-control {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border-color: rgba(255, 255, 255, 0.15) !important;
                border-radius: 8px;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            
            .Select-control:hover {
                border-color: rgba(102, 126, 234, 0.3) !important;
                background-color: rgba(255, 255, 255, 0.08) !important;
            }
            
            .Select-control:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
            }
            
            .Select-menu-outer {
                background-color: rgba(30, 33, 57, 0.95) !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-radius: 8px;
                backdrop-filter: blur(10px);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
            }
            
            .Select-option {
                background-color: transparent !important;
                color: #e8eaf6 !important;
                transition: all 0.2s ease;
            }
            
            .Select-option:hover {
                background-color: rgba(102, 126, 234, 0.2) !important;
                padding-left: 15px;
            }
            
            .Select-option.is-selected {
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.2)) !important;
                color: #a4a8ff !important;
            }
            
            .custom-dropdown .Select-value-label,
            .custom-dropdown .Select-placeholder,
            .custom-dropdown input {
                color: #e8eaf6 !important;
            }
            
            .custom-dropdown .Select-value {
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1)) !important;
                border-color: rgba(102, 126, 234, 0.3) !important;
                color: #e8eaf6 !important;
                border-radius: 6px;
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
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                border-bottom: 2px solid transparent;
            }
            
            .tab.active {
                border-bottom: 2px solid #667eea;
                color: #a4a8ff;
            }
            
            /* Animações de entrada */
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
            
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
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
            
            @keyframes glow {
                0%, 100% { 
                    box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
                }
                50% { 
                    box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
                }
            }
            
            @keyframes shimmer {
                0% { background-position: -1000px 0; }
                100% { background-position: 1000px 0; }
            }
            
            .js-plotly-plot {
                animation: fadeInUp 0.8s ease-out;
                border-radius: 12px;
            }
            
            ._dash-loading {
                position: relative;
            }
            
            ._dash-loading::after {
                content: "";
                position: absolute;
                top: 50%;
                left: 50%;
                width: 60px;
                height: 60px;
                margin: -30px 0 0 -30px;
                border: 4px solid rgba(102, 126, 234, 0.2);
                border-top: 4px solid #667eea;
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
            
            [id*="card-"]:nth-child(1) { animation-delay: 0.05s; }
            [id*="card-"]:nth-child(2) { animation-delay: 0.1s; }
            [id*="card-"]:nth-child(3) { animation-delay: 0.15s; }
            [id*="card-"]:nth-child(4) { animation-delay: 0.2s; }
            
            [id*="card-"]:hover {
                transform: translateY(-8px) scale(1.01);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            @keyframes skeletonLoading {
                0% { background-position: -200px 0; }
                100% { background-position: calc(200px + 100%) 0; }
            }
            
            .skeleton-loader {
                background: linear-gradient(
                    90deg,
                    rgba(102, 126, 234, 0.1) 0%,
                    rgba(102, 126, 234, 0.3) 50%,
                    rgba(102, 126, 234, 0.1) 100%
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
            
            .success-feedback { 
                animation: successPulse 1s ease-out;
            }
            
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
                background: linear-gradient(90deg, #667eea, #764ba2, #a855f7);
                animation: progressBar 2s ease-out;
            }
            
            /* Glassmorphism cards */
            .glass-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 12px;
                transition: all 0.3s ease;
            }
            
            .glass-card:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.25);
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
            }
            
            /* Badge styling */
            .badge-success {
                background: linear-gradient(135deg, #4ade80, #22c55e);
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                display: inline-block;
            }
            
            .badge-warning {
                background: linear-gradient(135deg, #fbbf24, #f59e0b);
                color: #0a0e27;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                display: inline-block;
            }
            
            .badge-error {
                background: linear-gradient(135deg, #f87171, #ef4444);
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                display: inline-block;
            }
            
            /* Dividers */
            .divider {
                height: 1px;
                background: linear-gradient(90deg, transparent, #667eea, transparent);
                margin: 20px 0;
            }
        </style>
        <script>
            // Load theme preference from localStorage on page load
            document.addEventListener('DOMContentLoaded', function() {
                const savedTheme = localStorage.getItem('nimbusvita-theme') || 'dark';
                const isDark = savedTheme === 'dark';
                // Store in sessionStorage for access by callbacks
                sessionStorage.setItem('nimbusvita-current-theme', savedTheme);
            });
            
            // Function to save theme preference
            function saveThemePreference(theme) {
                localStorage.setItem('nimbusvita-theme', theme);
                sessionStorage.setItem('nimbusvita-current-theme', theme);
            }
            
            // Expose saveThemePreference globally for use in callbacks
            window.saveThemePreference = saveThemePreference;
        </script>
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


def apply_plotly_template(fig: go.Figure, height: int = 500, show_legend: bool = True) -> go.Figure:
    """Apply consistent modern theme to Plotly figures."""
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        font=dict(
            family='Inter, sans-serif',
            size=12,
            color=COLORS['text']
        ),
        hovermode='x unified',
        showlegend=show_legend,
        legend=dict(
            bgcolor=f'rgba(30, 33, 57, 0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1,
            font=dict(size=11, color=COLORS['text']),
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
        ),
        margin=dict(t=40, b=40, l=60, r=40),
        height=height,
        title=dict(
            font=dict(size=18, color=COLORS['text'], family='Inter, sans-serif'),
            x=0.05,
            xanchor='left',
        ),
    )
    
    # Aplicar cores aos traces
    colors_cycle = ['#667eea', '#764ba2', '#4facfe', '#a4a8ff', '#14b8a6', '#f59e0b']
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'marker'):
            trace.marker.line.color = COLORS['border']
            trace.marker.line.width = 1
        # Usar cores do ciclo se não tiver cor específica
        if not hasattr(trace, 'line') or not trace.line.color:
            trace_color = colors_cycle[i % len(colors_cycle)]
            if hasattr(trace, 'marker'):
                trace.marker.color = trace_color
            if hasattr(trace, 'line'):
                trace.line.color = trace_color
    
    return fig


def error_figure(message: str, height: int = 400) -> go.Figure:
    """Create error/empty placeholder figure with message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS['text_secondary'], family='Inter, sans-serif')
    )
    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['card'],
        height=height,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

def _tab_style():
    """Return default tab style."""
    return {
        'color': COLORS['text_secondary'], 
        'backgroundColor': 'transparent',
        'border': 'none',
        'padding': '15px 30px',
        'fontSize': '1em',
        'fontWeight': '500',
        'transition': 'all 0.3s ease'
    }

def _tab_selected_style():
    """Return selected tab style."""
    return {
        'color': COLORS['accent'], 
        'backgroundColor': COLORS['card'], 
        'fontWeight': '600',
        'borderTop': f'3px solid {COLORS["accent"]}',
        'borderLeft': 'none',
        'borderRight': 'none',
        'borderBottom': 'none',
        'borderRadius': '8px 8px 0 0'
    }


__all__ = [
    'COLORS', 
    'INDEX_STRING', 
    'metrics_unavailable_figure', 
    'page_header', 
    'apply_plotly_template', 
    'error_figure'
]
