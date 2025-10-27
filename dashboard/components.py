"""Reusable UI components for the NimbusVita dashboard."""
from __future__ import annotations

from dash import html

from .core.theme import COLORS


def create_card(children, title: str | None = None) -> html.Div:
    """Return a styled card container used throughout the dashboard."""
    content = []
    if title:
        content.append(html.H3(title, style={
            'color': COLORS['text'],
            'marginBottom': '14px',
            'fontSize': '1.25em',
            'fontWeight': '700',
            'paddingBottom': '8px',
            'borderBottom': f'1px solid {COLORS["border"]}',
            # usar cor sólida para garantir legibilidade em temas escuros
        }))

    content.extend(children if isinstance(children, list) else [children])

    return html.Div(
        content,
        style={
            'backgroundColor': COLORS['card'],
            'padding': '25px',
            'borderRadius': '15px',
            'marginBottom': '25px',
            'boxShadow': '0 8px 32px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLORS["border"]}',
            'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
            'position': 'relative',
            'overflow': 'hidden',
        },
        className='card-hover',
    )


__all__ = ['create_card']
