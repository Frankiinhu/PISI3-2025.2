"""Reusable UI components for the NimbusVita dashboard."""
from __future__ import annotations

from dash import html

from .core.theme import COLORS


def create_card(children, title: str | None = None, gradient: bool = False, glass: bool = False) -> html.Div:
    """Return a styled card container with optional gradient or glassmorphism effect."""
    content = []
    
    # Adiciona barra de destaque gradiente no topo se houver título
    if title:
        content.append(html.Div([
            html.Div(style={
                'position': 'absolute',
                'top': '0',
                'left': '0',
                'right': '0',
                'height': '3px',
                'background': COLORS['gradient_accent'],
                'borderRadius': '15px 15px 0 0',
            }),
            html.H3(title, style={
                'color': COLORS['text'],
                'marginBottom': '14px',
                'marginTop': '8px',
                'fontSize': '1.25em',
                'fontWeight': '700',
                'paddingBottom': '12px',
                'borderBottom': f'2px solid {COLORS["border"]}',
                'display': 'flex',
                'alignItems': 'center',
                'gap': '10px',
            })
        ], style={'position': 'relative'}))

    content.extend(children if isinstance(children, list) else [children])
    
    # Estilo base
    card_style = {
        'padding': '25px',
        'borderRadius': '16px',
        'marginBottom': '25px',
        'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        'position': 'relative',
        'overflow': 'hidden',
    }
    
    # Aplica efeito glassmorphism
    if glass:
        card_style.update({
            'background': COLORS['glass_medium'],
            'backdropFilter': 'blur(10px)',
            'WebkitBackdropFilter': 'blur(10px)',
            'border': f'1px solid {COLORS["glass_border"]}',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.37)',
        })
    # Aplica gradiente
    elif gradient:
        card_style.update({
            'background': COLORS['gradient_card'],
            'border': f'1px solid {COLORS["border"]}',
            'boxShadow': '0 10px 40px rgba(102, 126, 234, 0.25)',
        })
    # Estilo padrão
    else:
        card_style.update({
            'backgroundColor': COLORS['card'],
            'border': f'1px solid {COLORS["border"]}',
            'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
        })

    return html.Div(
        content,
        style=card_style,
        className='card-hover',
    )


__all__ = ['create_card']
