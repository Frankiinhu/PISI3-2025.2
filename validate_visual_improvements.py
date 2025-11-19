#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Validação - Melhorias Visuais NimbusVita
Verifica se todas as novas funcionalidades estão funcionando corretamente
"""

import sys
import os

# Adicionar o caminho do projeto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_colors():
    """Teste 1: Verificar se as cores foram carregadas"""
    print("\n" + "="*70)
    print("✓ TESTE 1: Cores e Paleta")
    print("="*70)
    
    try:
        from dashboard.core.theme import COLORS
        
        # Verificar cores base
        assert 'primary' in COLORS, "Cor 'primary' não encontrada"
        assert 'text' in COLORS, "Cor 'text' não encontrada"
        
        # Verificar gradientes
        gradients = [k for k in COLORS.keys() if 'gradient' in k]
        assert len(gradients) >= 8, f"Esperado 8+ gradientes, encontrado {len(gradients)}"
        
        # Verificar glassmorphism
        assert 'glass_light' in COLORS, "Glassmorphism 'glass_light' não encontrada"
        assert 'glass_medium' in COLORS, "Glassmorphism 'glass_medium' não encontrada"
        assert 'glass_border' in COLORS, "Glassmorphism 'glass_border' não encontrada"
        
        print(f"✅ {len(COLORS)} cores carregadas com sucesso")
        print(f"✅ {len(gradients)} gradientes disponíveis")
        print(f"✅ Glassmorphism colors encontradas")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar cores: {e}")
        return False


def test_components():
    """Teste 2: Verificar se os componentes estão funcionando"""
    print("\n" + "="*70)
    print("✓ TESTE 2: Componentes")
    print("="*70)
    
    try:
        from dashboard.components import create_card
        from dash import html
        
        # Teste card padrão
        card1 = create_card(html.P("Teste"), title="Card Padrão")
        assert card1 is not None, "Card padrão não criado"
        
        # Teste card com gradiente
        card2 = create_card(html.P("Teste"), title="Card Gradiente", gradient=True)
        assert card2 is not None, "Card com gradiente não criado"
        
        # Teste card com glassmorphism
        card3 = create_card(html.P("Teste"), title="Card Glass", glass=True)
        assert card3 is not None, "Card com glassmorphism não criado"
        
        print("✅ Card padrão criado com sucesso")
        print("✅ Card com gradiente criado com sucesso")
        print("✅ Card com glassmorphism criado com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar componentes: {e}")
        return False


def test_kpi_cards():
    """Teste 3: Verificar se os KPI cards estão funcionando"""
    print("\n" + "="*70)
    print("✓ TESTE 3: KPI Cards")
    print("="*70)
    
    try:
        from dashboard.views.overview import _kpi_card
        from dashboard.core.theme import COLORS
        
        # Teste KPI com gradiente
        kpi1 = _kpi_card('📊', 'Teste', '123', COLORS['primary'], 'Descrição', 'gradient_blue')
        assert kpi1 is not None, "KPI card não criado"
        
        # Teste com diferentes gradientes
        gradients = ['gradient_primary', 'gradient_success', 'gradient_warning', 'gradient_error']
        for grad in gradients:
            kpi = _kpi_card('🎯', 'Teste', '100', COLORS['primary'], gradient=grad)
            assert kpi is not None, f"KPI com {grad} não criado"
        
        print("✅ KPI card padrão criado com sucesso")
        print(f"✅ {len(gradients)} variações de KPI cards criadas com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar KPI cards: {e}")
        return False


def test_plotly_template():
    """Teste 4: Verificar se o template de gráficos está funcionando"""
    print("\n" + "="*70)
    print("✓ TESTE 4: Template Plotly")
    print("="*70)
    
    try:
        from dashboard.core.theme import apply_plotly_template
        import plotly.graph_objects as go
        import pandas as pd
        
        # Criar figura simples
        df = pd.DataFrame({'X': [1, 2, 3], 'Y': [10, 20, 30]})
        fig = go.Figure(data=go.Scatter(x=df['X'], y=df['Y']))
        
        # Aplicar template
        fig_styled = apply_plotly_template(fig, height=500)
        
        assert fig_styled is not None, "Template não aplicado"
        assert fig_styled.layout.plot_bgcolor is not None, "Plot background não configurado"
        
        print("✅ Template de gráficos aplicado com sucesso")
        print("✅ Layout configurado corretamente")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar template Plotly: {e}")
        return False


def test_css_styles():
    """Teste 5: Verificar se o CSS foi carregado"""
    print("\n" + "="*70)
    print("✓ TESTE 5: CSS e Estilos")
    print("="*70)
    
    try:
        from dashboard.core.theme import INDEX_STRING
        
        # Verificar se INDEX_STRING contém estilos importantes
        assert 'fadeInUp' in INDEX_STRING, "Animação 'fadeInUp' não encontrada"
        assert 'glassmorphism' in INDEX_STRING or 'backdrop-filter' in INDEX_STRING, "Glassmorphism não encontrado"
        assert 'gradient' in INDEX_STRING, "Gradientes CSS não encontrados"
        assert 'animation' in INDEX_STRING or '@keyframes' in INDEX_STRING, "Animações não encontradas"
        
        # Contar animações
        keyframes_count = INDEX_STRING.count('@keyframes')
        
        print(f"✅ CSS de animações carregado ({keyframes_count}+ @keyframes)")
        print("✅ Glassmorphism CSS encontrado")
        print("✅ Gradientes CSS encontrados")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar CSS: {e}")
        return False


def test_imports():
    """Teste 6: Verificar se todos os imports funcionam"""
    print("\n" + "="*70)
    print("✓ TESTE 6: Imports")
    print("="*70)
    
    try:
        # Imports principais
        from dashboard.core.theme import COLORS, apply_plotly_template
        from dashboard.components import create_card
        from dashboard.views.overview import _kpi_card
        
        print("✅ COLORS importado com sucesso")
        print("✅ apply_plotly_template importado com sucesso")
        print("✅ create_card importado com sucesso")
        print("✅ _kpi_card importado com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao importar: {e}")
        return False


def test_data_loading():
    """Teste 7: Verificar se os dados carregam corretamente"""
    print("\n" + "="*70)
    print("✓ TESTE 7: Carregamento de Dados")
    print("="*70)
    
    try:
        from dashboard.core.data_context import get_context
        
        ctx = get_context()
        assert ctx is not None, "Context não carregado"
        assert hasattr(ctx, 'df'), "DataFrame não encontrado no context"
        assert len(ctx.df) > 0, "DataFrame vazio"
        
        print(f"✅ Context carregado com sucesso")
        print(f"✅ {len(ctx.df)} linhas de dados carregadas")
        print(f"✅ {len(ctx.df.columns)} colunas disponíveis")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return False


def run_all_tests():
    """Executar todos os testes"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "VALIDAÇÃO DE MELHORIAS VISUAIS" + " "*23 + "║")
    print("║" + " "*15 + "NimbusVita Dashboard" + " "*33 + "║")
    print("╚" + "═"*68 + "╝")
    
    tests = [
        ("Cores e Paleta", test_colors),
        ("Componentes", test_components),
        ("KPI Cards", test_kpi_cards),
        ("Template Plotly", test_plotly_template),
        ("CSS e Estilos", test_css_styles),
        ("Imports", test_imports),
        ("Carregamento de Dados", test_data_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro não esperado em {name}: {e}")
            results.append((name, False))
    
    # Sumário
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status}: {name}")
    
    print("="*70)
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODAS AS VALIDAÇÕES PASSARAM!")
        print("✨ As melhorias visuais estão prontas para uso!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} teste(s) falharam")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
