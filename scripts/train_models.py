"""
Script para treinar modelos de Machine Learning
"""
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import DataLoader
from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_classifier(data_path: str, save_path: str):
    """
    Treina modelo de classificação
    
    Args:
        data_path: Caminho para o dataset
        save_path: Caminho para salvar o modelo
    """
    logger.info("="*50)
    logger.info("TREINAMENTO DO CLASSIFICADOR")
    logger.info("="*50)
    
    # Carregar dados
    logger.info(f"Carregando dados de: {data_path}")
    loader = DataLoader(data_path)
    df = loader.get_clean_data()
    
    # Inicializar classificador
    classifier = DiagnosisClassifier(random_state=42)
    
    # Preparar dados
    logger.info("Preparando dados para treinamento...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=0.2)
    
    # Treinar modelo
    logger.info("Treinando Random Forest...")
    classifier.train_random_forest(
        X_train, y_train,
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Avaliar modelo
    logger.info("Avaliando modelo...")
    metrics = classifier.evaluate(X_test, y_test)
    
    logger.info(f"\nMétricas do Modelo:")
    logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Validação cruzada
    logger.info("\nRealizando validação cruzada...")
    cv_results = classifier.cross_validate(X_train, y_train, cv=5)
    logger.info(f"CV Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    
    # Importância das features
    logger.info("\nTop 10 Features Mais Importantes:")
    top_features = classifier.get_feature_importance(top_n=10)
    for feature, importance in top_features.items():
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Atribuir métricas ao objeto classifier e salvar modelo
    classifier.metrics = metrics
    # Salvar modelo
    logger.info(f"\nSalvando modelo em: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classifier.save_model(save_path)

    # Salvar métricas em results/
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, 'classifier_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    logger.info(f"Métricas do classificador salvas em: {metrics_path}")
    
    logger.info("✅ Treinamento do classificador concluído!")
    return classifier, metrics


def train_clusterer(data_path: str, save_path: str):
    """
    Treina modelo de clusterização
    
    Args:
        data_path: Caminho para o dataset
        save_path: Caminho para salvar o modelo
    """
    logger.info("\n" + "="*50)
    logger.info("TREINAMENTO DO CLUSTERIZADOR")
    logger.info("="*50)
    
    # Carregar dados
    logger.info(f"Carregando dados de: {data_path}")
    loader = DataLoader(data_path)
    df = loader.get_clean_data()
    
    # Inicializar clusterizador
    clusterer = DiseaseClusterer(random_state=42)
    
    # Preparar dados
    logger.info("Preparando dados para clusterização...")
    X_scaled = clusterer.prepare_data(df, exclude_cols=['Diagnóstico'])
    
    # Encontrar número ótimo de clusters
    logger.info("Encontrando número ótimo de clusters...")
    optimal_results = clusterer.find_optimal_clusters(X_scaled, max_clusters=10)
    
    best_k = optimal_results['k_values'][
        optimal_results['silhouette_scores'].index(max(optimal_results['silhouette_scores']))
    ]
    logger.info(f"Melhor número de clusters: {best_k}")
    
    # Treinar K-Means
    logger.info(f"Treinando K-Means com {best_k} clusters...")
    clusterer.train_kmeans(X_scaled, n_clusters=best_k)
    
    # Avaliar clusterização
    logger.info("Avaliando clusterização...")
    metrics = clusterer.evaluate(X_scaled)
    
    logger.info(f"\nMétricas de Clusterização:")
    logger.info(f"Número de Clusters: {metrics['n_clusters']}")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
    logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")
    
    # Perfis dos clusters
    logger.info("\nCriando perfis dos clusters...")
    cluster_profiles = clusterer.get_cluster_profiles(df, X_scaled)
    logger.info(f"Perfis criados para {len(cluster_profiles)} clusters")
    
    # Salvar modelo
    logger.info(f"\nSalvando modelo em: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    clusterer.save_model(save_path)

    # Salvar métricas e perfis em results/
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    cluster_metrics_path = os.path.join(results_dir, 'cluster_metrics.json')
    with open(cluster_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Métricas de clusterização salvas em: {cluster_metrics_path}")

    # Salvar perfis dos clusters em CSV
    cluster_profiles = clusterer.get_cluster_profiles(df, X_scaled)
    profiles_path = os.path.join(results_dir, 'cluster_profiles.csv')
    cluster_profiles.to_csv(profiles_path)
    logger.info(f"Perfis de cluster salvos em: {profiles_path}")
    
    logger.info("✅ Treinamento do clusterizador concluído!")
    return clusterer, metrics


def main():
    """Função principal"""
    # Definir caminhos
    DATA_PATH = '../data/DATASET FINAL WRDP.csv'  # Ajuste conforme necessário
    CLASSIFIER_SAVE_PATH = '../models/saved_models/classifier_model.pkl'
    CLUSTERER_SAVE_PATH = '../models/saved_models/clustering_model.pkl'
    
    try:
        # Treinar classificador
        classifier, clf_metrics = train_classifier(DATA_PATH, CLASSIFIER_SAVE_PATH)
        
        # Treinar clusterizador
        clusterer, cluster_metrics = train_clusterer(DATA_PATH, CLUSTERER_SAVE_PATH)
        
        logger.info("\n" + "="*50)
        logger.info("🎉 TREINAMENTO COMPLETO!")
        logger.info("="*50)
        logger.info(f"Classificador salvo em: {CLASSIFIER_SAVE_PATH}")
        logger.info(f"Clusterizador salvo em: {CLUSTERER_SAVE_PATH}")
        
    except Exception as e:
        logger.error(f"❌ Erro durante o treinamento: {e}")
        raise


if __name__ == '__main__':
    main()
