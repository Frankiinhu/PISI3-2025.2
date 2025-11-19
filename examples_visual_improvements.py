"""
Exemplo de uso das melhorias visuais do dashboard NimbusVita
"""

from dashboard.components import create_card
from dashboard.core.theme import COLORS, apply_plotly_template
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

# ============================================================================
# EXEMPLO 1: Usando Cards com Gradientes
# ============================================================================

def example_cards():
    """Exemplos de cards com diferentes estilos"""
    
    return html.Div([
        html.H2("Exemplos de Cards", style={'color': COLORS['text'], 'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                create_card(
                    html.P("Card padrão com estilo base"),
                    title="Card Padrão"
                )
            ], md=6),
            
            dbc.Col([
                create_card(
                    html.P("Card com gradiente vibrant"),
                    title="Card com Gradiente",
                    gradient=True
                )
            ], md=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_card(
                    html.P("Card com efeito glassmorphism e blur"),
                    title="Card Glass",
                    glass=True
                )
            ], md=6),
            
            dbc.Col([
                create_card([
                    html.H4("Conteúdo Complexo", style={'color': COLORS['text']}),
                    html.P("Você pode adicionar qualquer conteúdo aqui"),
                    html.Hr(),
                    html.Span("Badge ", className='badge-success'),
                    html.Span("exemplo", className='badge-warning'),
                ], title="Card com Conteúdo")
            ], md=6),
        ])
    ])


# ============================================================================
# EXEMPLO 2: Usando KPI Cards com Gradientes
# ============================================================================

def example_kpi_cards():
    """Exemplos de KPI cards com diferentes gradientes"""
    from dashboard.views.overview import _kpi_card
    
    return html.Div([
        html.H2("KPI Cards com Gradientes", style={'color': COLORS['text'], 'marginBottom': '30px'}),
        
        dbc.Row([
            _kpi_card('📊', 'Métrica 1', '1,234', COLORS['accent'], 'Descrição', 'gradient_blue'),
            _kpi_card('🚀', 'Métrica 2', '56.2%', COLORS['success'], 'Crescimento', 'gradient_success'),
            _kpi_card('⚠️', 'Métrica 3', '89', COLORS['warning'], 'Alertas', 'gradient_warning'),
            _kpi_card('🎯', 'Métrica 4', '98%', COLORS['info'], 'Performance', 'gradient_teal'),
            _kpi_card('✅', 'Métrica 5', '42k', COLORS['primary'], 'Confirmadas', 'gradient_primary'),
        ])
    ])


# ============================================================================
# EXEMPLO 3: Usando Template de Gráficos
# ============================================================================

def example_plotly_charts():
    """Exemplos de gráficos com template consistente"""
    
    import pandas as pd
    
    # Criar dados de exemplo
    df = pd.DataFrame({
        'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun'],
        'Vendas': [150, 230, 200, 280, 220, 310],
        'Custos': [100, 120, 110, 140, 130, 160],
        'Lucro': [50, 110, 90, 140, 90, 150]
    })
    
    # ---- GRÁFICO 1: Linha ----
    fig_line = px.line(
        df, 
        x='Mês', 
        y=['Vendas', 'Custos'],
        title='Série Temporal',
        markers=True,
        labels={'value': 'Valor (R$)', 'variable': 'Tipo'}
    )
    fig_line = apply_plotly_template(fig_line, height=400)
    
    # ---- GRÁFICO 2: Barra ----
    fig_bar = px.bar(
        df,
        x='Mês',
        y='Lucro',
        title='Lucro Mensal',
        color='Lucro',
        color_continuous_scale=['#f87171', '#fbbf24', '#4ade80']
    )
    fig_bar = apply_plotly_template(fig_bar, height=400)
    
    # ---- GRÁFICO 3: Scatter ----
    df_scatter = pd.DataFrame({
        'X': [1, 2, 3, 4, 5, 6, 7, 8],
        'Y': [10, 15, 13, 17, 20, 25, 22, 28],
        'Cluster': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
    })
    
    fig_scatter = px.scatter(
        df_scatter,
        x='X',
        y='Y',
        color='Cluster',
        size='Y',
        title='Análise de Clusters'
    )
    fig_scatter = apply_plotly_template(fig_scatter, height=400)
    
    return html.Div([
        html.H2("Gráficos com Template Consistente", 
                style={'color': COLORS['text'], 'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_line)
            ], md=6),
            dbc.Col([
                dcc.Graph(figure=fig_bar)
            ], md=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_scatter)
            ], md=12),
        ])
    ])


# ============================================================================
# EXEMPLO 4: Usando Cores e Glassmorphism
# ============================================================================

def example_colors_and_glass():
    """Exemplos de uso da paleta de cores e glassmorphism"""
    
    return html.Div([
        html.H2("Paleta de Cores e Glassmorphism", 
                style={'color': COLORS['text'], 'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("Background Gradient", style={'color': COLORS['text']}),
                    html.P("Gradiente linear como fundo")
                ], style={
                    'background': COLORS['gradient_primary'],
                    'padding': '30px',
                    'borderRadius': '12px',
                    'color': 'white',
                    'textAlign': 'center'
                })
            ], md=6, style={'marginBottom': '20px'}),
            
            dbc.Col([
                html.Div([
                    html.H5("Glassmorphism Effect", style={'color': COLORS['text']}),
                    html.P("Com blur e semi-transparência")
                ], style={
                    'background': COLORS['glass_medium'],
                    'backdropFilter': 'blur(10px)',
                    'WebkitBackdropFilter': 'blur(10px)',
                    'border': f'1px solid {COLORS["glass_border"]}',
                    'padding': '30px',
                    'borderRadius': '12px',
                    'color': COLORS['text'],
                    'textAlign': 'center'
                })
            ], md=6, style={'marginBottom': '20px'}),
        ]),
        
        # Exemplo de cores semânticas
        dbc.Row([
            dbc.Col([
                html.Div("✅ Success", className='badge-success', style={'fontSize': '1.1em', 'padding': '15px'})
            ], md=3),
            dbc.Col([
                html.Div("⚠️ Warning", className='badge-warning', style={'fontSize': '1.1em', 'padding': '15px'})
            ], md=3),
            dbc.Col([
                html.Div("❌ Error", className='badge-error', style={'fontSize': '1.1em', 'padding': '15px'})
            ], md=3),
            dbc.Col([
                html.Div("ℹ️ Info", style={
                    'background': f'linear-gradient(135deg, {COLORS["info"]} 0%, {COLORS["info_light"]} 100%)',
                    'color': 'white',
                    'padding': '15px',
                    'borderRadius': '20px',
                    'fontSize': '1.1em',
                    'textAlign': 'center',
                    'fontWeight': '600'
                })
            ], md=3),
        ])
    ])


# ============================================================================
# EXEMPLO 5: Integrando Tudo
# ============================================================================

def complete_example():
    """Exemplo completo mostrando todas as melhorias integradas"""
    
    return html.Div([
        html.H1("Dashboard com Melhorias Visuais", 
                style={'color': COLORS['text'], 'marginBottom': '40px'}),
        
        create_card([
            html.P("Este é um exemplo completo mostrando:")
        ], title="Bem-vindo"),
        
        example_kpi_cards(),
        
        html.Hr(style={'borderColor': COLORS['border'], 'margin': '40px 0'}),
        
        example_cards(),
        
        html.Hr(style={'borderColor': COLORS['border'], 'margin': '40px 0'}),
        
        example_plotly_charts(),
        
        html.Hr(style={'borderColor': COLORS['border'], 'margin': '40px 0'}),
        
        example_colors_and_glass(),
        
        html.Div(style={'height': '50px'})  # Spacing
    ], style={'padding': '30px'})


if __name__ == '__main__':
    """
    Para usar em seu dashboard, adicione isso ao seu app.py ou views:
    
    from examples_visual_improvements import complete_example
    
    app.layout = complete_example()
    """
    print("✅ Arquivo de exemplos carregado com sucesso!")
    print("\nPara usar os exemplos, importe as funções no seu dashboard:")
    print("  from examples_visual_improvements import complete_example")
    print("\nOu importe funções específicas:")
    print("  from examples_visual_improvements import example_kpi_cards")
    print("  from examples_visual_improvements import example_plotly_charts")
