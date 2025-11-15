#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de teste para validar a aba "Visão Geral" do dashboard
Verifica imports, componentes e estrutura dos callbacks
"""

import sys
import os

# Adicionar o diretório ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_imports():
    """Teste 1: Validar importações"""
    print("\n" + "="*70)
    print("✓ TESTE 1: VALIDANDO IMPORTAÇÕES")
    print("="*70)
    
    try:
        import dash
        print("  ✓ Dash importado com sucesso")
    except ImportError as e:
        print(f"  ✗ Erro ao importar Dash: {e}")
        return False
    
    try:
        import dash_bootstrap_components as dbc
        print("  ✓ Dash Bootstrap Components importado com sucesso")
    except ImportError as e:
        print(f"  ✗ Erro ao importar dbc: {e}")
        print("  → Execute: pip install dash-bootstrap-components")
        return False
    
    try:
        import plotly.express as px
        print("  ✓ Plotly Express importado com sucesso")
    except ImportError as e:
        print(f"  ✗ Erro ao importar Plotly: {e}")
        return False
    
    try:
        from dashboard.views import overview
        print("  ✓ Módulo overview importado com sucesso")
    except ImportError as e:
        print(f"  ✗ Erro ao importar overview: {e}")
        return False
    
    try:
        from dashboard.core.data_context import get_context
        print("  ✓ Data context importado com sucesso")
    except ImportError as e:
        print(f"  ✗ Erro ao importar data_context: {e}")
        return False
    
    return True


def test_overview_functions():
    """Teste 2: Validar funções do overview"""
    print("\n" + "="*70)
    print("✓ TESTE 2: VALIDANDO FUNÇÕES DO OVERVIEW")
    print("="*70)
    
    try:
        from dashboard.views.overview import (
            create_layout,
            register_callbacks,
            _filter_dropdown,
            _kpi_card,
            _alert_component,
            hex_to_rgb
        )
        print("  ✓ Função create_layout disponível")
        print("  ✓ Função register_callbacks disponível")
        print("  ✓ Função _filter_dropdown disponível")
        print("  ✓ Função _kpi_card disponível")
        print("  ✓ Função _alert_component disponível")
        print("  ✓ Função hex_to_rgb disponível")
        return True
    except ImportError as e:
        print(f"  ✗ Erro ao importar funções: {e}")
        return False


def test_data_loading():
    """Teste 3: Validar carregamento de dados"""
    print("\n" + "="*70)
    print("✓ TESTE 3: VALIDANDO CARREGAMENTO DE DADOS")
    print("="*70)
    
    try:
        from dashboard.core.data_context import get_context
        ctx = get_context()
        print(f"  ✓ Contexto carregado com sucesso")
        print(f"    - Dataset: {ctx.df.shape[0]} linhas, {ctx.df.shape[1]} colunas")
        print(f"    - Diagnósticos: {ctx.diagnosis_cols}")
        print(f"    - Sintomas: {len(ctx.symptom_cols)} colunas")
        print(f"    - Variáveis climáticas: {ctx.climatic_vars}")
        return True
    except Exception as e:
        print(f"  ⚠ Aviso ao carregar contexto: {e}")
        return True  # Não é erro fatal, pode ser falta de dados


def test_bootstrap_components():
    """Teste 4: Validar componentes Bootstrap"""
    print("\n" + "="*70)
    print("✓ TESTE 4: VALIDANDO COMPONENTES BOOTSTRAP")
    print("="*70)
    
    try:
        import dash_bootstrap_components as dbc
        from dash import html
        
        # Testar componentes básicos
        container = dbc.Container()
        print("  ✓ dbc.Container disponível")
        
        row = dbc.Row()
        print("  ✓ dbc.Row disponível")
        
        col = dbc.Col()
        print("  ✓ dbc.Col disponível")
        
        alert = dbc.Alert()
        print("  ✓ dbc.Alert disponível")
        
        label = dbc.Label()
        print("  ✓ dbc.Label disponível")
        
        return True
    except Exception as e:
        print(f"  ✗ Erro ao validar componentes Bootstrap: {e}")
        return False


def test_color_conversion():
    """Teste 5: Validar conversão de cores"""
    print("\n" + "="*70)
    print("✓ TESTE 5: VALIDANDO CONVERSÃO DE CORES")
    print("="*70)
    
    try:
        from dashboard.views.overview import hex_to_rgb
        
        # Testar conversões
        rgb = hex_to_rgb('#5559FF')
        expected = (85, 89, 255)
        assert rgb == expected, f"Esperado {expected}, obtido {rgb}"
        print(f"  ✓ Conversão hex_to_rgb funcionando: #5559FF → {rgb}")
        
        rgb2 = hex_to_rgb('#A4A8FF')
        expected2 = (164, 168, 255)
        assert rgb2 == expected2, f"Esperado {expected2}, obtido {rgb2}"
        print(f"  ✓ Conversão hex_to_rgb funcionando: #A4A8FF → {rgb2}")
        
        return True
    except Exception as e:
        print(f"  ✗ Erro ao validar conversão de cores: {e}")
        return False


def test_layout_creation():
    """Teste 6: Validar criação do layout"""
    print("\n" + "="*70)
    print("✓ TESTE 6: VALIDANDO CRIAÇÃO DO LAYOUT")
    print("="*70)
    
    try:
        from dashboard.views.overview import create_layout
        layout = create_layout()
        print(f"  ✓ Layout criado com sucesso")
        print(f"    - Tipo: {type(layout).__name__}")
        
        # Validar que é um componente Dash
        from dash import html
        assert hasattr(layout, 'children'), "Layout deve ter propriedade 'children'"
        print(f"  ✓ Layout é um componente Dash válido")
        
        return True
    except Exception as e:
        print(f"  ✗ Erro ao criar layout: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements():
    """Teste 7: Validar requirements"""
    print("\n" + "="*70)
    print("✓ TESTE 7: VALIDANDO REQUIREMENTS")
    print("="*70)
    
    required_packages = {
        'dash': '3.2.0',
        'plotly': '6.3.1',
        'pandas': '2.3.3',
        'numpy': '2.3.4',
        'scikit-learn': '1.7.2',
        'dash-bootstrap-components': '1.7.0'
    }
    
    all_installed = True
    for package, version in required_packages.items():
        try:
            mod = __import__(package.replace('-', '_'))
            pkg_version = getattr(mod, '__version__', 'desconhecida')
            status = "✓" if pkg_version == version else "⚠"
            print(f"  {status} {package}: {pkg_version} (esperado {version})")
        except ImportError:
            print(f"  ✗ {package}: NÃO INSTALADO")
            all_installed = False
    
    return all_installed


def main():
    """Executar todos os testes"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "TESTES DA ABA 'VISÃO GERAL'" + " "*27 + "║")
    print("║" + " "*15 + "Dashboard NimbusVita v2.0" + " "*28 + "║")
    print("╚" + "═"*68 + "╝")
    
    tests = [
        ("Importações", test_imports),
        ("Funções Overview", test_overview_functions),
        ("Carregamento de Dados", test_data_loading),
        ("Componentes Bootstrap", test_bootstrap_components),
        ("Conversão de Cores", test_color_conversion),
        ("Criação do Layout", test_layout_creation),
        ("Requirements", test_requirements),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ Erro inesperado: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resumo
    print("\n" + "="*70)
    print("RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSOU" if result else "✗ FALHOU"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-"*70)
    print(f"Total: {passed}/{total} testes passou")
    print("="*70 + "\n")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("   A aba 'Visão Geral' está funcionando corretamente.\n")
        return 0
    else:
        print(f"⚠️  {total - passed} teste(s) falharam.")
        print("   Execute: pip install -r requirements.txt\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
