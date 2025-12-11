"""Data and model loading helpers for the NimbusVita dashboard."""
from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure src package is discoverable when this module is imported standalone.
_SYS_INSERTED = False
if 'src' not in sys.modules:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    _SYS_INSERTED = True

from src.data_processing.data_loader import DataLoader
from src.data_processing.eda import ExploratoryDataAnalysis
from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer


@dataclass
class DataContext:
    """In-memory resources shared across the dashboard."""

    df: pd.DataFrame
    eda: ExploratoryDataAnalysis
    classifier: DiagnosisClassifier
    clusterer: DiseaseClusterer
    loader: DataLoader
    symptom_cols: List[str]
    diagnosis_cols: List[str]
    climatic_vars: List[str]


_context: Optional[DataContext] = None


def load_data_context() -> DataContext:
    """Load datasets and models once and cache the result."""
    global _context
    if _context is not None:
        return _context

    logger.info("Carregando dados...")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'DATASET FINAL WRDP.csv')
    loader = DataLoader(data_path)
    df_global = loader.get_clean_data()
    eda_global = ExploratoryDataAnalysis(df_global)

    feature_dict = loader.get_feature_names()
    symptom_cols = feature_dict.get('symptoms', [])
    diagnosis_cols = feature_dict.get('diagnosis', []) or [feature_dict.get('target', 'Diagnóstico')]
    climatic_vars = feature_dict.get('climatic', [])

    logger.info("Carregando modelos...")
    classifier = DiagnosisClassifier()
    clusterer = DiseaseClusterer()

    # Tentar carregar do diretório models/ na raiz do projeto (onde train_models.py salva)
    try:
        # Tentar primeiro em models/saved_models/classifier.joblib
        classifier_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved_models', 'classifier.joblib')
        if not os.path.exists(classifier_path):
            # Se não encontrar, tentar em models/classifier.joblib
            classifier_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'classifier.joblib')
        
        if os.path.exists(classifier_path):
            logger.info(f"Carregando classificador de: {classifier_path}")
            classifier.load_model(classifier_path)
            logger.info(f"Classificador carregado com sucesso. Modelo: {type(classifier.model).__name__ if classifier.model else 'None'}")
        else:
            logger.warning(f"Arquivo de classificador não encontrado em: {classifier_path}")
    except Exception as exc:
        logger.error(f"Erro ao carregar classificador: {exc}", exc_info=True)

    try:
        # Tentar primeiro em models/saved_models/clusterer.joblib
        clusterer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved_models', 'clusterer.joblib')
        if not os.path.exists(clusterer_path):
            # Se não encontrar, tentar em models/clusterer.joblib
            clusterer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'clusterer.joblib')
        
        if os.path.exists(clusterer_path):
            logger.info(f"Carregando clusterizador de: {clusterer_path}")
            clusterer.load_model(clusterer_path)
            logger.info(f"Clusterizador carregado com sucesso. Modo: {clusterer.mode}, Features: {len(clusterer.feature_names) if clusterer.feature_names else 0}")
        else:
            logger.warning(f"Arquivo de clusterizador não encontrado em: {clusterer_path}")
    except Exception as exc:
        logger.error(f"Erro ao carregar clusterizador: {exc}", exc_info=True)

    _context = DataContext(
        df=df_global,
        eda=eda_global,
        classifier=classifier,
        clusterer=clusterer,
        loader=loader,
        symptom_cols=symptom_cols,
        diagnosis_cols=diagnosis_cols,
        climatic_vars=climatic_vars,
    )
    return _context


def get_context() -> DataContext:
    """Public accessor that guarantees resources are loaded."""
    return load_data_context()


def clear_context() -> None:
    """Drop cached resources (useful for testing)."""
    global _context
    _context = None


def is_classifier_available() -> bool:
    ctx = get_context()
    return ctx.classifier is not None and getattr(ctx.classifier, 'model', None) is not None


def has_feature_importances() -> bool:
    ctx = get_context()
    feature_importances = getattr(ctx.classifier, 'feature_importances', None)
    return feature_importances is not None


def get_cluster_feature_frame() -> pd.DataFrame:
    ctx = get_context()
    if getattr(ctx.clusterer, 'feature_names', None) is None:
        raise ValueError('Clusterizador indisponível ou sem feature names; execute o treinamento e carregue o modelo salvo.')

    # Check if using K-Modes (can handle categorical) or K-Means (numeric only)
    mode = getattr(ctx.clusterer, 'mode', 'kmeans')
    
    if mode == 'kmodes':
        # K-Modes can use all features (categorical and numeric)
        missing = [col for col in ctx.clusterer.feature_names if col not in ctx.df.columns]
        if missing:
            raise ValueError(
                'Colunas ausentes no dataset atual para reconstruir os clusters: ' + ', '.join(missing)
            )
        return ctx.df.loc[:, ctx.clusterer.feature_names]
    else:
        # K-Means needs numeric features only
        numeric_df = ctx.df.select_dtypes(include=[np.number])
        missing = [col for col in ctx.clusterer.feature_names if col not in numeric_df.columns]
        if missing:
            raise ValueError(
                'Colunas ausentes no dataset atual para reconstruir os clusters: ' + ', '.join(missing)
            )
        return numeric_df.loc[:, ctx.clusterer.feature_names]


def get_cluster_features_and_labels() -> Tuple[pd.DataFrame, pd.Series]:
    ctx = get_context()
    if getattr(ctx.clusterer, 'model', None) is None:
        raise ValueError('Clusterizador não carregado; execute o treinamento antes de gerar visualizações.')

    feature_frame = get_cluster_feature_frame()
    try:
        labels = ctx.clusterer.predict_cluster(feature_frame.values)
    except ValueError as exc:
        raise ValueError(f'Não foi possível gerar previsões de cluster: {exc}') from exc

    return feature_frame, pd.Series(labels, index=feature_frame.index, name='Cluster')


__all__ = [
    'DataContext',
    'clear_context',
    'get_context',
    'get_cluster_feature_frame',
    'get_cluster_features_and_labels',
    'has_feature_importances',
    'is_classifier_available',
    'load_data_context',
]
