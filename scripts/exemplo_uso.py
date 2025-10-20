"""
Exemplo de uso dos modelos treinados
"""
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import DataLoader
from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer


def exemplo_predicao_simples():
    """Exemplo de predição simples de diagnóstico"""
    print("="*60)
    print("EXEMPLO 1: Predição Simples de Diagnóstico")
    print("="*60)
    
    # Carregar modelo treinado
    classifier = DiagnosisClassifier()
    classifier.load_model('../models/saved_models/classifier_model.pkl')
    
    # Criar exemplo de paciente
    paciente = pd.DataFrame({
        'Idade': [35],
        'Gênero': [1],
        'Temperatura (°C)': [38.5],
        'Umidade': [75],
        'Velocidade do Vento (km/h)': [15],
        'Febre': [1],
        'Tosse': [1],
        'Fadiga': [1],
        'Dor de Cabeça': [1],
        'Dor no Peito': [0],
        # Adicione outros sintomas com valor 0 se necessário
    })
    
    # Garantir que todas as features estão presentes
    for feature in classifier.feature_names:
        if feature not in paciente.columns:
            paciente[feature] = 0
    
    # Reordenar colunas
    paciente = paciente[classifier.feature_names]
    
    # Fazer predição
    diagnostico = classifier.predict(paciente)[0]
    probabilidades = classifier.predict_proba(paciente)[0]
    
    print(f"\n📋 Dados do Paciente:")
    print(f"  Idade: 35 anos")
    print(f"  Temperatura: 38.5°C")
    print(f"  Umidade: 75%")
    print(f"  Sintomas: Febre, Tosse, Fadiga, Dor de Cabeça")
    
    print(f"\n🎯 Diagnóstico Predito: {diagnostico}")
    print(f"📊 Confiança: {max(probabilidades)*100:.2f}%")
    
    print(f"\n📈 Probabilidades por Diagnóstico:")
    for classe, prob in zip(classifier.label_encoder.classes_, probabilidades):
        print(f"  {classe}: {prob*100:.2f}%")


def exemplo_identificacao_cluster():
    """Exemplo de identificação de cluster e fatores de risco"""
    print("\n" + "="*60)
    print("EXEMPLO 2: Identificação de Cluster e Fatores de Risco")
    print("="*60)
    
    # Carregar modelo de clusterização
    clusterer = DiseaseClusterer()
    clusterer.load_model('../models/saved_models/clustering_model.pkl')
    
    # Carregar dados para análise
    loader = DataLoader('../data/DATASET FINAL WRDP.csv')
    df = loader.get_clean_data()
    
    # Preparar dados
    X_scaled = clusterer.prepare_data(df, exclude_cols=['Diagnóstico'])
    
    # Exemplo de paciente
    paciente_idx = 0
    cluster_id = clusterer.labels_[paciente_idx]
    
    print(f"\n🔍 Paciente #{paciente_idx}")
    print(f"  Cluster Identificado: {cluster_id}")
    
    # Identificar fatores de risco do cluster
    risk_factors = clusterer.identify_risk_factors(df, cluster_id)
    
    print(f"\n⚠️ Top 5 Fatores de Risco do Cluster {cluster_id}:")
    for i, (factor, data) in enumerate(list(risk_factors.items())[:5], 1):
        print(f"  {i}. {factor}")
        print(f"     Média no cluster: {data['cluster_mean']:.2f}")
        print(f"     Diferença: {data['relative_diff']:+.1f}%")


def exemplo_analise_eda():
    """Exemplo de análise exploratória de dados"""
    print("\n" + "="*60)
    print("EXEMPLO 3: Análise Exploratória de Dados")
    print("="*60)
    
    # Carregar dados
    loader = DataLoader('../data/DATASET FINAL WRDP.csv')
    df = loader.get_clean_data()
    
    # Criar objeto EDA
    from src.data_processing.eda import ExploratoryDataAnalysis
    eda = ExploratoryDataAnalysis(df)
    
    # Informações básicas
    info = eda.basic_info()
    print(f"\n📊 Informações do Dataset:")
    print(f"  Linhas: {info['shape'][0]}")
    print(f"  Colunas: {info['shape'][1]}")
    print(f"  Valores duplicados: {info['duplicates']}")
    
    # Distribuição de diagnósticos
    diag_dist = eda.categorical_distribution('Diagnóstico')
    print(f"\n🏥 Top 5 Diagnósticos Mais Frequentes:")
    for i, (diag, count) in enumerate(diag_dist.head().iterrows(), 1):
        print(f"  {i}. {diag}: {count['Count']} casos ({count['Percentage']:.1f}%)")
    
    # Estatísticas climáticas
    climate_vars = ['Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)']
    stats = eda.numerical_statistics(climate_vars)
    print(f"\n🌡️ Estatísticas Climáticas:")
    print(stats.round(2))


def exemplo_api_request():
    """Exemplo de requisição para a API"""
    print("\n" + "="*60)
    print("EXEMPLO 4: Requisição para API (necessita API rodando)")
    print("="*60)
    
    print("""
    # Para fazer uma requisição à API, use:
    
    import requests
    import json
    
    url = "http://localhost:5000/predict"
    
    data = {
        "idade": 35,
        "temperatura": 38.5,
        "umidade": 75,
        "velocidade_vento": 15,
        "sintomas": {
            "Febre": 1,
            "Tosse": 1,
            "Fadiga": 1,
            "Dor de Cabeça": 1
        }
    }
    
    response = requests.post(url, json=data)
    resultado = response.json()
    
    print(f"Diagnóstico: {resultado['diagnostico_predito']}")
    print(f"Confiança: {resultado['confianca']*100:.2f}%")
    
    # Para verificar o status da API:
    health = requests.get("http://localhost:5000/health")
    print(health.json())
    """)


if __name__ == '__main__':
    print("\n🌡️ VITANIMBUS - EXEMPLOS DE USO\n")
    
    try:
        exemplo_predicao_simples()
        exemplo_identificacao_cluster()
        exemplo_analise_eda()
        exemplo_api_request()
        
        print("\n" + "="*60)
        print("✅ Todos os exemplos executados com sucesso!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: Arquivo não encontrado - {e}")
        print("   Certifique-se de que os modelos foram treinados e o dataset está disponível.")
        print("   Execute: python scripts/train_models.py")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
