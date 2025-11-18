"""
Módulo de Clusterização para Identificação de Padrões
"""
import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DiseaseClusterer:
    """Classe para clusterização de padrões de doenças"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.labels_: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def prepare_data(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> np.ndarray:
        if exclude_cols is None:
            exclude_cols = []
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        for col in exclude_cols:
            if col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[col])
        self.feature_names = numeric_df.columns.tolist()
        X_scaled = self.scaler.fit_transform(numeric_df)
        logger.info(f"Dados preparados para clusterização: {X_scaled.shape}")
        return X_scaled

    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10, sample_size: int = 1000) -> Dict:
        ks = list(range(2, max_clusters + 1))
        silhouettes = []
        inertias = []
        
        # Use sampling for silhouette score if dataset is large
        use_sampling = len(X) > sample_size
        X_sample = None
        sample_indices = None
        
        if use_sampling:
            logger.info(f"Using sample of {sample_size} points for silhouette calculation (dataset has {len(X)} points)")
            rng = np.random.RandomState(self.random_state)
            sample_indices = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
            X_sample = X[sample_indices]
        
        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            km.fit(X)
            inertias.append(float(km.inertia_))
            labels = km.labels_
            
            # Silhouette só é válido se > 1 cluster e < n amostras
            if len(np.unique(labels)) > 1:
                try:
                    if use_sampling and X_sample is not None and sample_indices is not None:
                        # Calculate silhouette on sample
                        sample_labels = labels[sample_indices]
                        silhouettes.append(float(silhouette_score(X_sample, sample_labels)))
                    else:
                        silhouettes.append(float(silhouette_score(X, labels)))
                except Exception as e:
                    logger.warning(f"Error calculating silhouette for k={k}: {e}")
                    silhouettes.append(float('nan'))
            else:
                silhouettes.append(float('nan'))

        # Heurística simples para "cotovelo" via segunda diferença
        elbow_k = None
        if len(inertias) >= 3:
            second_diff = np.diff(inertias, n=2)
            idx = int(np.argmin(second_diff)) + 2  # mapeia para k em ks
            elbow_k = ks[idx] if 0 <= idx < len(ks) else None

        silhouette_best_k = None
        if any(np.isfinite(s) for s in silhouettes):
            silhouette_best_k = ks[int(np.nanargmax(np.array(silhouettes)))]

        return {
            'ks': ks,
            'silhouette_scores': silhouettes,
            'inertias': inertias,
            'silhouette_best_k': silhouette_best_k,
            'elbow_k': elbow_k
        }

    def train_kmeans(self, X: np.ndarray, n_clusters: int = 3, **kwargs) -> None:
        params = dict(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
        params.update(kwargs)
        self.model = KMeans(**params)
        self.model.fit(X)
        self.labels_ = self.model.labels_

    def evaluate(self, X: np.ndarray, sample_size: int = 1000) -> Dict:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        labels = self.model.labels_
        n_clusters = len(np.unique(labels))
        metrics = {'n_clusters': int(n_clusters)}
        
        if n_clusters > 1:
            # Use sampling for silhouette if dataset is large
            if len(X) > sample_size:
                logger.info(f"Using sample of {sample_size} points for silhouette calculation")
                rng = np.random.RandomState(self.random_state)
                sample_indices = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
                X_sample = X[sample_indices]
                sample_labels = labels[sample_indices]
                metrics['silhouette_score'] = float(silhouette_score(X_sample, sample_labels))
            else:
                metrics['silhouette_score'] = float(silhouette_score(X, labels))
            
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
        else:
            metrics['silhouette_score'] = float('nan')
            metrics['davies_bouldin_score'] = float('nan')
            metrics['calinski_harabasz_score'] = float('nan')
        return metrics

    def reduce_dimensions(self, X: np.ndarray, n_components: int = 3) -> np.ndarray:
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        return self.pca.fit_transform(X)

    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
        }
        joblib.dump(payload, filepath)
        logger.info(f"Modelo de clusterização salvo em: {filepath}")

    def load_model(self, filepath: str) -> None:
        payload = joblib.load(filepath)
        self.model = payload['model']
        self.scaler = payload['scaler']
        self.pca = payload.get('pca')
        self.feature_names = payload.get('feature_names')
        self.random_state = payload.get('random_state', self.random_state)
        self.labels_ = getattr(self.model, 'labels_', None)
        logger.info(f"Modelo de clusterização carregado de: {filepath}")