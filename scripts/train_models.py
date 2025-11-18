"""
Script para treinar modelos de Machine Learning
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.data_loader import DataLoader
from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_classifier(data_path: Path | str, save_path: Path | str, use_smote: bool = False):
    """
    Treina modelo de classificação
    
    Args:
        data_path: Caminho para o dataset
        save_path: Caminho para salvar o modelo
        use_smote: Se True, aplica SMOTE para balanceamento de classes
    """
    logger.info("=" * 50)
    logger.info("TREINAMENTO DO CLASSIFICADOR")
    if use_smote:
        logger.info("⚖️  MODO: COM SMOTE (Balanceamento de Classes)")
    else:
        logger.info("📊 MODO: BASE (Sem Balanceamento)")
    logger.info("=" * 50)

    data_path = Path(data_path)
    save_path = Path(save_path)

    logger.info(f"Carregando dados de: {data_path}")
    loader = DataLoader(str(data_path))
    df = loader.get_clean_data()

    classifier = DiagnosisClassifier(random_state=42)

    logger.info("Preparando dados para treinamento...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(df, test_size=0.2)

    if use_smote:
        logger.info("Treinando Random Forest com SMOTE...")
        classifier.train_with_smote(
            X_train, y_train,
            model_type='random_forest',
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
    else:
        logger.info("Treinando Random Forest...")
        classifier.train_random_forest(
            X_train, y_train,
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )

    logger.info("Avaliando modelo...")
    metrics = classifier.evaluate(X_test, y_test)

    logger.info("\n📈 Métricas do Modelo:")
    logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
    logger.info(f"Acurácia Balanceada: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    logger.info(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")

    logger.info("\n🔄 Realizando validação cruzada...")
    cv_results = classifier.cross_validate(X_train, y_train, cv=5)
    logger.info(f"CV Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

    logger.info("\n⭐ Top 10 Features Mais Importantes:")
    try:
        top_features = classifier.get_feature_importance(top_n=10)
        for feature, importance in top_features.items():
            logger.info(f"  {feature}: {importance:.4f}")
    except AttributeError:
        logger.info("  Importância de features não disponível para este modelo.")

    logger.info("\n🔍 Calculando SHAP values...")
    try:
        classifier.calculate_shap_values(X_test, max_samples=100)
        logger.info("  ✓ SHAP values calculados e salvos com o modelo")
    except Exception as e:
        logger.warning(f"  ⚠️  Erro ao calcular SHAP values: {e}")

    classifier.metrics = metrics

    logger.info(f"\n💾 Salvando modelo em: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(save_path))

    logger.info("✅ Treinamento do classificador concluído!")
    return classifier, metrics


def train_clusterer(data_path: Path | str, save_path: Path | str):
    """
    Treina modelo de clusterização
    
    Args:
        data_path: Caminho para o dataset
        save_path: Caminho para salvar o modelo
    """
    logger.info("\n" + "=" * 50)
    logger.info("TREINAMENTO DO CLUSTERIZADOR")
    logger.info("=" * 50)

    data_path = Path(data_path)
    save_path = Path(save_path)

    logger.info(f"Carregando dados de: {data_path}")
    loader = DataLoader(str(data_path))
    df = loader.get_clean_data()

    clusterer = DiseaseClusterer(random_state=42)

    logger.info("Preparando dados para clusterização...")
    X_scaled = clusterer.prepare_data(df, exclude_cols=['Diagnóstico'])

    logger.info("Encontrando número ótimo de clusters...")
    optimal_results = clusterer.find_optimal_clusters(X_scaled, max_clusters=10)

    silhouette_best_k = optimal_results.get('silhouette_best_k')
    elbow_k = optimal_results.get('elbow_k')

    if silhouette_best_k is not None:
        logger.info(f"Melhor número de clusters (silhouette): {silhouette_best_k}")
    if elbow_k is not None and elbow_k != silhouette_best_k:
        logger.info(f"Sugestão pelo método do cotovelo: {elbow_k}")

    best_k = silhouette_best_k or elbow_k or 3

    logger.info(f"Treinando K-Means com {best_k} clusters...")
    clusterer.train_kmeans(X_scaled, n_clusters=best_k)

    logger.info("Avaliando clusterização...")
    metrics = clusterer.evaluate(X_scaled)

    logger.info("\nMétricas de Clusterização:")
    logger.info(f"Número de Clusters: {metrics['n_clusters']}")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
    logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f}")

    logger.info(f"\nSalvando modelo em: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    clusterer.save_model(str(save_path))

    logger.info("✅ Treinamento do clusterizador concluído!")
    return clusterer, metrics


def main():
    parser = argparse.ArgumentParser(description="Treinar modelos de classificação e clusterização.")
    parser.add_argument('--data', required=True, help='Caminho para o dataset (CSV ou fonte esperada pelo DataLoader)')
    parser.add_argument('--out-dir', default='models', help='Diretório para salvar os modelos')
    parser.add_argument('--classifier-name', default='classifier.joblib')
    parser.add_argument('--clusterer-name', default='clusterer.joblib')
    parser.add_argument('--skip-classifier', action='store_true', help='Pular treinamento do classificador')
    parser.add_argument('--skip-clusterer', action='store_true', help='Pular treinamento do clusterizador')
    parser.add_argument('--use-smote', action='store_true', help='Aplicar SMOTE para balanceamento de classes no classificador')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_classifier:
        train_classifier(args.data, out_dir / args.classifier_name, use_smote=args.use_smote)

    if not args.skip_clusterer:
        train_clusterer(args.data, out_dir / args.clusterer_name)


if __name__ == '__main__':
    main()
