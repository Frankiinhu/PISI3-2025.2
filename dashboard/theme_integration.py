"""Integration module for theme system in Dash app."""
from typing import Dict
from dash import html, dcc, Input, Output, callback, clientside_callback
import dash_bootstrap_components as dbc
from dashboard.core.theme_manager import ThemeManager, DARK_THEME, LIGHT_THEME
from dashboard.core.theme import COLORS


def create_theme_header_content() -> html.Div:
    """Create the header content with theme toggle button."""
    return html.Div(
        style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'width': '100%'},
        children=[
            html.Div(style={'flex': 1}, children=[
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
            ]),
            html.Div(
                style={
                    'position': 'absolute',
                    'right': '30px',
                    'top': '50%',
                    'transform': 'translateY(-50%)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '10px'
                },
                children=[
                    dbc.Button(
                        id='theme-toggle-btn',
                        className='theme-toggle-btn',
                        style={
                            'width': '50px',
                            'height': '50px',
                            'borderRadius': '50%',
                            'border': '2px solid rgba(255,255,255,0.3)',
                            'backgroundColor': 'rgba(255,255,255,0.1)',
                            'color': 'white',
                            'fontSize': '1.3em',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s ease',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'padding': '0',
                            'hover': {
                                'backgroundColor': 'rgba(255,255,255,0.2)',
                                'borderColor': 'rgba(255,255,255,0.5)'
                            }
                        },
                        children='🌙'
                    ),
                    # Hidden store to track current theme
                    dcc.Store(id='theme-store', data={'theme': 'dark'})
                ]
            )
        ]
    )


def create_theme_switching_callback(app) -> None:
    """Register the theme switching callback."""
    
    @app.callback(
        [Output('theme-toggle-btn', 'children'),
         Output('theme-store', 'data')],
        [Input('theme-toggle-btn', 'n_clicks')],
        prevent_initial_call=False
    )
    def toggle_theme(n_clicks):
        """Toggle theme between dark and light."""
        # Get current theme from localStorage via dcc.Store
        theme = ThemeManager.toggle_theme()
        theme_name = ThemeManager.get_theme_name()
        icon = ThemeManager.get_theme_icon()
        
        return [icon, {'theme': theme}]


def get_dynamic_styles(theme_data: Dict) -> Dict:
    """Get dynamic styles based on theme."""
    theme_name = theme_data.get('theme', 'dark') if isinstance(theme_data, dict) else 'dark'
    colors = ThemeManager.get_theme(theme_name)
    
    return {
        'backgroundColor': colors['background'],
        'color': colors['text'],
        'borderColor': colors['border']
    }


def create_theme_persistence_script() -> html.Script:
    """Create a script for persisting theme to localStorage."""
    script = """
    <script>
        // Load theme preference from localStorage on page load
        document.addEventListener('DOMContentLoaded', function() {
            const savedTheme = localStorage.getItem('nimbusvita-theme') || 'dark';
            const themeStore = document.getElementById('theme-store');
            if (themeStore) {
                themeStore.setAttribute('data-theme', savedTheme);
            }
            applyTheme(savedTheme);
        });
        
        // Save theme preference when changed
        function saveTheme(theme) {
            localStorage.setItem('nimbusvita-theme', theme);
        }
        
        // Apply theme styles
        function applyTheme(theme) {
            const isDark = theme === 'dark';
            const root = document.documentElement;
            
            if (isDark) {
                root.style.setProperty('--theme-bg', '#0a0e27');
                root.style.setProperty('--theme-text', '#e8eaf6');
                root.style.setProperty('--theme-card', '#1e2139');
            } else {
                root.style.setProperty('--theme-bg', '#f8f9fa');
                root.style.setProperty('--theme-text', '#1a1a1a');
                root.style.setProperty('--theme-card', '#ffffff');
            }
        }
    </script>
    """
    return html.Script(script)


def get_theme_aware_background(theme: str = 'dark') -> str:
    """Get gradient background based on theme."""
    if theme == 'dark':
        return f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["background_light"]} 100%)'
    else:
        colors = ThemeManager.get_theme('light')
        return f'linear-gradient(135deg, {colors["background"]} 0%, {colors["background_light"]} 100%)'


def get_theme_aware_header_gradient(theme: str = 'dark') -> str:
    """Get header gradient based on theme."""
    if theme == 'dark':
        return f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_light"]} 100%)'
    else:
        colors = ThemeManager.get_theme('light')
        return f'linear-gradient(135deg, {colors["primary"]} 0%, {colors["primary_light"]} 100%)'
