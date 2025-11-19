"""
Theme Manager - Gerencia temas Dark/Light para o NimbusVita Dashboard
Centraliza paletas de cores e fornece funções para alternância de tema
"""
from __future__ import annotations
from typing import Dict, Literal

# ============================================================================
# PALETA DARK THEME (Original)
# ============================================================================
DARK_THEME: Dict[str, str] = {
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
    'gradient_primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_success': 'linear-gradient(135deg, #4ade80 0%, #22c55e 100%)',
    'gradient_warning': 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
    'gradient_error': 'linear-gradient(135deg, #f87171 0%, #ef4444 100%)',
    'gradient_accent': 'linear-gradient(135deg, #667eea 0%, #4facfe 100%)',
    'gradient_purple': 'linear-gradient(135deg, #a855f7 0%, #8b5cf6 100%)',
    'gradient_blue': 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
    'gradient_teal': 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
    'gradient_card': 'linear-gradient(135deg, #1e2139 0%, #252a48 100%)',
    'glass_light': 'rgba(255, 255, 255, 0.05)',
    'glass_medium': 'rgba(255, 255, 255, 0.1)',
    'glass_border': 'rgba(255, 255, 255, 0.18)',
    'info': '#60a5fa',
    'info_light': '#93c5fd',
    'hover_overlay': 'rgba(102, 126, 234, 0.1)',
    'active_overlay': 'rgba(102, 126, 234, 0.2)',
}

# ============================================================================
# PALETA LIGHT THEME (Novo)
# ============================================================================
LIGHT_THEME: Dict[str, str] = {
    'background': '#f8f9fa',
    'background_light': '#ffffff',
    'primary': '#4f46e5',
    'primary_light': '#6366f1',
    'primary_dark': '#4338ca',
    'secondary': '#e5e7eb',
    'secondary_light': '#f59e0b',
    'secondary_dark': '#d97706',
    'text': '#1f2937',
    'text_secondary': '#6b7280',
    'accent': '#8b5cf6',
    'accent_light': '#a78bfa',
    'accent_secondary': '#0ea5e9',
    'card': '#ffffff',
    'card_hover': '#f3f4f6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'border': '#d1d5db',
    'gradient_primary': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
    'gradient_success': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    'gradient_warning': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
    'gradient_error': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
    'gradient_accent': 'linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%)',
    'gradient_purple': 'linear-gradient(135deg, #a855f7 0%, #7c3aed 100%)',
    'gradient_blue': 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
    'gradient_teal': 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
    'gradient_card': 'linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)',
    'glass_light': 'rgba(0, 0, 0, 0.03)',
    'glass_medium': 'rgba(0, 0, 0, 0.05)',
    'glass_border': 'rgba(0, 0, 0, 0.1)',
    'info': '#3b82f6',
    'info_light': '#60a5fa',
    'hover_overlay': 'rgba(79, 70, 229, 0.08)',
    'active_overlay': 'rgba(79, 70, 229, 0.15)',
}

# ============================================================================
# GERENCIADOR DE TEMAS
# ============================================================================
class ThemeManager:
    """Gerenciador centralizado de temas"""
    
    _current_theme: Literal['dark', 'light'] = 'dark'
    _themes = {
        'dark': DARK_THEME,
        'light': LIGHT_THEME,
    }
    
    @classmethod
    def get_theme(cls, theme: Literal['dark', 'light'] | None = None) -> Dict[str, str]:
        """Retorna a paleta de cores do tema especificado ou atual"""
        if theme is None:
            theme = cls._current_theme
        return cls._themes.get(theme, cls._themes['dark'])
    
    @classmethod
    def set_theme(cls, theme: Literal['dark', 'light']) -> Dict[str, str]:
        """Define o tema atual e retorna sua paleta"""
        if theme in cls._themes:
            cls._current_theme = theme
        return cls.get_theme(theme)
    
    @classmethod
    def get_current_theme(cls) -> Literal['dark', 'light']:
        """Retorna o tema atual"""
        return cls._current_theme
    
    @classmethod
    def toggle_theme(cls) -> Dict[str, str]:
        """Alterna entre dark e light theme"""
        new_theme = 'light' if cls._current_theme == 'dark' else 'dark'
        return cls.set_theme(new_theme)
    
    @classmethod
    def get_theme_name(cls) -> str:
        """Retorna o nome do tema atual em português"""
        return 'Escuro' if cls._current_theme == 'dark' else 'Claro'
    
    @classmethod
    def get_theme_icon(cls) -> str:
        """Retorna um ícone para o tema atual"""
        return '🌙' if cls._current_theme == 'dark' else '☀️'


def get_colors(theme: Literal['dark', 'light'] | None = None) -> Dict[str, str]:
    """Função auxiliar para obter cores do tema"""
    return ThemeManager.get_theme(theme)


def apply_theme_to_css(theme: Literal['dark', 'light']) -> str:
    """Gera CSS dinâmico baseado no tema"""
    colors = ThemeManager.get_theme(theme)
    
    if theme == 'light':
        return f"""
        :root {{
            --bg-primary: {colors['background']};
            --bg-card: {colors['card']};
            --text-primary: {colors['text']};
            --text-secondary: {colors['text_secondary']};
            --border: {colors['border']};
        }}
        
        body {{
            background: linear-gradient(135deg, {colors['background']} 0%, {colors['background_light']} 100%);
            color: {colors['text']};
            transition: all 0.3s ease;
        }}
        
        input[type="number"],
        input[type="text"],
        input[type="date"],
        select {{
            background-color: {colors['card']} !important;
            border: 1px solid {colors['border']} !important;
            color: {colors['text']} !important;
        }}
        
        input[type="number"]:focus,
        input[type="text"]:focus,
        input[type="date"]:focus,
        select:focus {{
            border-color: {colors['primary']} !important;
            box-shadow: 0 0 0 3px rgba({colors['primary']}, 0.1) !important;
        }}
        
        .Select-control {{
            background-color: {colors['card']} !important;
            border-color: {colors['border']} !important;
        }}
        
        .Select-menu-outer {{
            background-color: {colors['card']} !important;
            border: 1px solid {colors['border']} !important;
        }}
        
        .Select-option:hover {{
            background-color: {colors['background_light']} !important;
        }}
        """
    else:
        return """/* Dark theme CSS (original) */"""


__all__ = ['ThemeManager', 'DARK_THEME', 'LIGHT_THEME', 'get_colors', 'apply_theme_to_css']
