"""
Theme Toggle Component - Componente interativo para trocar de tema
"""
from dash import html, dcc, callback, Input, Output, clientside_callback
import dash_bootstrap_components as dbc
from dashboard.core.theme_manager import ThemeManager
from dashboard.core.theme import COLORS


def create_theme_toggle() -> html.Div:
    """
    Cria um botão de toggle para alternar entre dark e light theme
    """
    current_theme = ThemeManager.get_current_theme()
    icon = '☀️' if current_theme == 'dark' else '🌙'
    tooltip_text = 'Mudar para tema claro' if current_theme == 'dark' else 'Mudar para tema escuro'
    
    return html.Div([
        dbc.Button(
            icon,
            id='theme-toggle-button',
            className='theme-toggle-btn',
            n_clicks=0,
            title=tooltip_text,
            style={
                'backgroundColor': 'transparent',
                'border': f'2px solid {COLORS["border"]}',
                'color': COLORS['text'],
                'borderRadius': '50%',
                'width': '45px',
                'height': '45px',
                'fontSize': '20px',
                'cursor': 'pointer',
                'transition': 'all 0.3s ease',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'fontWeight': 'bold',
            }
        ),
        dcc.Store(id='theme-store', data={'theme': current_theme}),
    ], style={'display': 'inline-block'})


def get_theme_toggle_callback():
    """
    Retorna o callback para o toggle de tema
    Deve ser adicionado ao app.py
    """
    @callback(
        [Output('theme-store', 'data'),
         Output('theme-toggle-button', 'children'),
         Output('theme-toggle-button', 'title'),
         Output('_pages_content', 'style', allow_duplicate=True)],
        Input('theme-toggle-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def toggle_theme(n_clicks):
        """Alterna o tema e atualiza a UI"""
        colors = ThemeManager.toggle_theme()
        current = ThemeManager.get_current_theme()
        
        icon = '☀️' if current == 'dark' else '🌙'
        tooltip = 'Mudar para tema claro' if current == 'dark' else 'Mudar para tema escuro'
        
        # Atualizar o estilo da página
        page_style = {
            'backgroundColor': colors['background'],
            'color': colors['text'],
            'transition': 'all 0.3s ease',
        }
        
        return (
            {'theme': current},
            icon,
            tooltip,
            page_style
        )


__all__ = ['create_theme_toggle', 'get_theme_toggle_callback']
