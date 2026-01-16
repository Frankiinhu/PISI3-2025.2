"""Shared UI helpers for NimbusVita dashboard views."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.core.theme import COLORS
from dashboard.components import create_card


def section_header(title: str, subtitle: str | None = None, accent: str = 'accent') -> html.Div:
    """Create section header with consistent styling."""
    return html.Div([
        html.H3(
            title,
            style={
                'color': COLORS['text'],
                'marginBottom': '10px',
                'fontSize': '1.7em',
                'fontWeight': '700',
                'borderLeft': f'6px solid {COLORS[accent]}',
                'paddingLeft': '14px',
                'background': 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%)',
            },
        ),
        html.P(
            subtitle,
            style={
                'color': COLORS['text_secondary'],
                'fontSize': '1em',
                'marginBottom': '24px',
                'paddingLeft': '18px',
            },
        ) if subtitle else None,
    ])


def graph_row(cards: Sequence[html.Div]) -> html.Div:
    """Create a responsive row of graph cards."""
    return html.Div(list(cards), style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '20px',
        'marginBottom': '20px',
    })


def graph_card(graph_id: str, title: str, flex: str = '1 1 360px') -> html.Div:
    """Create a card with a graph component."""
    return html.Div(
        create_card([dcc.Graph(id=graph_id)], title),
        style={'flex': flex, 'minWidth': '320px'},
    )


def filter_dropdown(
    component_id: str,
    label: str,
    options: Iterable[dict[str, Any]],
    value,
    width: str = '25%'
) -> html.Div:
    """Create filter dropdown for flex layout sections."""
    return html.Div([
        html.Label(label, style={'color': COLORS['text'], 'fontWeight': '600', 'display': 'block', 'marginBottom': '8px'}),
        dcc.Dropdown(
            id=component_id,
            options=list(options),  # type: ignore[arg-type]
            value=value,
            clearable=False,
            className='custom-dropdown',
            style={'backgroundColor': COLORS['secondary']},
        ),
    ], style={'flex': f'1 1 {width}', 'minWidth': '220px'})


def filter_dropdown_col(
    component_id: str,
    label: str,
    options: Iterable[dict[str, Any]],
    value
) -> dbc.Col:
    """Create filter dropdown using Bootstrap column layout."""
    return dbc.Col([
        dbc.Label(label, html_for=component_id, style={
            'color': COLORS['text'],
            'fontWeight': '600',
            'marginBottom': '8px',
            'fontSize': '0.95em',
        }),
        dcc.Dropdown(
            id=component_id,
            options=list(options),  # type: ignore[arg-type]
            value=value,
            clearable=False,
            className='custom-dropdown',
            style={
                'backgroundColor': COLORS['secondary'],
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["border"]}',
            },
        ),
    ], md=6, sm=12)


def kpi_card(icon: str, label: str, value: str, subtitle: str = '', gradient: str = 'gradient_primary') -> dbc.Col:
    """Create a KPI card with gradient and modern effects."""
    gradient_map = {
        'gradient_primary': COLORS.get('gradient_primary', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'),
        'gradient_success': COLORS.get('gradient_success', 'linear-gradient(135deg, #4ade80 0%, #22c55e 100%)'),
        'gradient_warning': COLORS.get('gradient_warning', 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)'),
        'gradient_error': COLORS.get('gradient_error', 'linear-gradient(135deg, #f87171 0%, #ef4444 100%)'),
        'gradient_teal': COLORS.get('gradient_teal', 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)'),
        'gradient_blue': COLORS.get('gradient_blue', 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'),
    }

    selected_gradient = gradient_map.get(gradient, gradient_map['gradient_primary'])

    return dbc.Col([
        html.Div([
            html.Div(style={
                'position': 'absolute',
                'top': '0',
                'left': '0',
                'right': '0',
                'height': '4px',
                'width': '100%',
                'background': selected_gradient,
                'borderRadius': '12px 12px 0 0',
            }),
            html.Div([
                html.Div(icon, style={
                    'fontSize': '3em',
                    'marginBottom': '12px',
                    'textAlign': 'center',
                    'filter': 'drop-shadow(0 0 10px rgba(102, 126, 234, 0.4))',
                    'transition': 'all 0.3s ease',
                })
            ], style={
                'background': 'rgba(102, 126, 234, 0.1)',
                'padding': '12px',
                'borderRadius': '12px',
                'marginBottom': '12px',
                'backdropFilter': 'blur(10px)',
            }),
            html.H6(label, style={
                'color': COLORS['text_secondary'],
                'margin': '0',
                'fontSize': '0.8em',
                'fontWeight': '600',
                'textTransform': 'uppercase',
                'letterSpacing': '1px',
                'textAlign': 'center',
            }),
            html.H3(value, style={
                'background': selected_gradient,
                '-webkit-background-clip': 'text',
                '-webkit-text-fill-color': 'transparent',
                'backgroundClip': 'text',
                'margin': '12px 0 0 0',
                'fontSize': '2.2em',
                'fontWeight': '800',
                'textAlign': 'center',
                'letterSpacing': '-0.5px',
            }),
            html.P(subtitle, style={
                'color': COLORS['text_secondary'],
                'margin': '8px 0 0 0',
                'fontSize': '0.75em',
                'textAlign': 'center',
                'fontWeight': '500',
                'opacity': '0.8',
            }) if subtitle else None,
        ], style={
            'background': f'linear-gradient(135deg, {COLORS["card"]} 0%, {COLORS["card_hover"]} 100%)',
            'padding': '28px 20px',
            'borderRadius': '12px',
            'textAlign': 'center',
            'boxShadow': '0 12px 40px rgba(0,0,0,0.3)',
            'border': f'1px solid {COLORS["border"]}',
            'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            'height': '100%',
            'position': 'relative',
            'overflow': 'hidden',
        }, className='stat-card')
    ], md=6, lg=2, sm=6, xs=12, style={'marginBottom': '20px'})


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    values = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return values[0], values[1], values[2]


def alert_component(alert_type: str, title: str, message: str) -> dbc.Alert:
    """Create a consistent alert component."""
    alert_colors = {
        'warning': COLORS['warning'],
        'danger': COLORS['error'],
        'success': COLORS['success'],
        'info': COLORS['accent'],
    }

    icons = {
        'warning': '⚠️',
        'danger': '❌',
        'success': '✅',
        'info': 'ℹ️',
    }

    return dbc.Alert([
        html.Div([
            html.Span(icons.get(alert_type, 'ℹ️'), style={
                'fontSize': '1.3em',
                'marginRight': '12px',
                'display': 'inline-block',
            }),
            html.Div([
                html.Strong(title, style={'display': 'block', 'marginBottom': '4px'}),
                html.Span(message, style={'fontSize': '0.95em'}),
            ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
        ])
    ], color=alert_type, style={
        'backgroundColor': f'rgba({_hex_to_rgb(alert_colors[alert_type])}, 0.15)',
        'borderLeft': f'4px solid {alert_colors[alert_type]}',
        'borderRadius': '8px',
        'padding': '16px',
        'marginBottom': '16px',
        'borderTop': 'none',
        'borderRight': 'none',
        'borderBottom': 'none',
    }, className='alert-custom')


__all__ = [
    'section_header',
    'graph_row',
    'graph_card',
    'filter_dropdown',
    'filter_dropdown_col',
    'kpi_card',
    'alert_component',
]
