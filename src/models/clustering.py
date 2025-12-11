"""
Módulo de Clusterização para Identificação de Padrões
Suporta K-Means (dados numéricos) e K-Modes (dados categóricos/binários)
"""
import logging
from typing import Dict, List, Optional, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

try:
    from kmodes.kmodes import KModes
    KMODES_AVAILABLE = True
except ImportError:
    KMODES_AVAILABLE = False
    logging.warning("kmodes não disponível. Instale com: pip install kmodes")

logger = logging.getLogger(__name__)


class DiseaseClusterer:
    """Classe para clusterização de padrões de doenças
    
    Suporta dois modos:
    - 'kmeans': Para dados numéricos (padrão)
    - 'kmodes': Para dados categóricos/binários (melhor para datasets com muitas features binárias)
    """

    def __init__(self, random_state: int = 42, mode: Literal['kmeans', 'kmodes'] = 'kmodes'):
        self.random_state = random_state
        self.mode = mode
        self.model: Optional[KMeans | KModes] = None
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.labels_: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.categorical_cols: Optional[List[str]] = None
        self.numeric_cols: Optional[List[str]] = None

    def prepare_data(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> np.ndarray:
        """Prepara dados para clusterização baseado no modo (kmeans ou kmodes).
        
        Args:
            df: DataFrame com os dados
            exclude_cols: Colunas a excluir
            
        Returns:
            Array preparado para o algoritmo de clustering
        """
        if exclude_cols is None:
            exclude_cols = []
        
        if self.mode == 'kmodes':
            # Para K-Modes: usa todas as colunas (categóricas e numéricas)
            # Converte numéricas em categóricas via binning se necessário
            working_df = df.copy()
            for col in exclude_cols:
                if col in working_df.columns:
                    working_df = working_df.drop(columns=[col])
            
            self.feature_names = working_df.columns.tolist()
            
            # Identifica colunas categóricas e numéricas
            self.categorical_cols = working_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            self.numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Para K-Modes, converte tudo para categórico
            # Colunas numéricas são mantidas como estão (K-Modes pode lidar com elas)
            X_prepared = working_df.values
            logger.info(f"Dados preparados para K-Modes: {X_prepared.shape} ({len(self.categorical_cols)} categóricas, {len(self.numeric_cols)} numéricas)")
            return X_prepared
        
        else:  # kmeans
            # Para K-Means: apenas colunas numéricas, escaladas
            numeric_df = df.select_dtypes(include=[np.number]).copy()
            for col in exclude_cols:
                if col in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=[col])
            
            self.feature_names = numeric_df.columns.tolist()
            X_scaled = self.scaler.fit_transform(numeric_df)
            logger.info(f"Dados preparados para K-Means: {X_scaled.shape}")
            return X_scaled

    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10, sample_size: int = 5200) -> Dict:
        """Encontra número ótimo de clusters usando métodos elbow e silhouette.
        
        Args:
            X: Dados preparados
            max_clusters: Número máximo de clusters a testar
            sample_size: Tamanho da amostra para cálculo de silhouette
            
        Returns:
            Dict com resultados: ks, silhouette_scores, inertias/costs, silhouette_best_k, elbow_k
        """
        ks = list(range(2, max_clusters + 1))
        silhouettes = []
        costs = []  # inertia para KMeans, cost para KModes
        
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
            if self.mode == 'kmodes' and KMODES_AVAILABLE:
                # K-Modes para dados categóricos/binários
                km = KModes(n_clusters=k, init='Huang', random_state=self.random_state, n_init=5, verbose=0)
                try:
                    km.fit(X)
                    costs.append(float(km.cost_))
                    labels = km.labels_
                except Exception as e:
                    logger.warning(f"Error fitting K-Modes for k={k}: {e}")
                    costs.append(float('inf'))
                    continue
            else:
                # K-Means para dados numéricos
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
                km.fit(X)
                costs.append(float(km.inertia_))
                labels = km.labels_
            
            # Silhouette só é válido se > 1 cluster e < n amostras
            if len(np.unique(labels)) > 1:
                try:
                    if use_sampling and X_sample is not None and sample_indices is not None:
                        # Calculate silhouette on sample
                        sample_labels = labels[sample_indices]
                        
                        # Para K-Modes com dados categóricos, usar métrica hamming
                        if self.mode == 'kmodes':
                            # Silhouette com métrica hamming para dados categóricos
                            silhouettes.append(float(silhouette_score(X_sample, sample_labels, metric='hamming')))
                        else:
                            silhouettes.append(float(silhouette_score(X_sample, sample_labels)))
                    else:
                        if self.mode == 'kmodes':
                            silhouettes.append(float(silhouette_score(X, labels, metric='hamming')))
                        else:
                            silhouettes.append(float(silhouette_score(X, labels)))
                except Exception as e:
                    logger.warning(f"Error calculating silhouette for k={k}: {e}")
                    silhouettes.append(float('nan'))
            else:
                silhouettes.append(float('nan'))

        # Heurística simples para "cotovelo" via segunda diferença
        elbow_k = None
        if len(costs) >= 3:
            finite_costs = [c for c in costs if np.isfinite(c)]
            if len(finite_costs) >= 3:
                second_diff = np.diff(finite_costs, n=2)
                idx = int(np.argmin(second_diff)) + 2  # mapeia para k em ks
                elbow_k = ks[idx] if 0 <= idx < len(ks) else None

        silhouette_best_k = None
        if any(np.isfinite(s) for s in silhouettes):
            silhouette_best_k = ks[int(np.nanargmax(np.array(silhouettes)))]

        return {
            'ks': ks,
            'silhouette_scores': silhouettes,
            'inertias': costs,  # Mantém nome 'inertias' para compatibilidade
            'costs': costs,
            'silhouette_best_k': silhouette_best_k,
            'elbow_k': elbow_k
        }

    def train(self, X: np.ndarray, n_clusters: int = 3, **kwargs) -> None:
        """Treina o modelo de clustering baseado no modo configurado.
        
        Args:
            X: Dados preparados
            n_clusters: Número de clusters
            **kwargs: Parâmetros adicionais para o algoritmo
        """
        if self.mode == 'kmodes' and KMODES_AVAILABLE:
            params = dict(
                n_clusters=n_clusters, 
                init='Huang',
                random_state=self.random_state,
                n_init=5,
                verbose=0
            )
            params.update(kwargs)
            self.model = KModes(**params)
            self.model.fit(X)
            self.labels_ = self.model.labels_
            logger.info(f"K-Modes treinado com {n_clusters} clusters. Cost: {self.model.cost_:.2f}")
        else:
            params = dict(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
            params.update(kwargs)
            self.model = KMeans(**params)
            self.model.fit(X)
            self.labels_ = self.model.labels_
            logger.info(f"K-Means treinado com {n_clusters} clusters. Inertia: {self.model.inertia_:.2f}")
    
    def train_kmeans(self, X: np.ndarray, n_clusters: int = 3, **kwargs) -> None:
        """Método legado - use train() ao invés."""
        logger.warning("train_kmeans() está deprecado. Use train() ao invés.")
        old_mode = self.mode
        self.mode = 'kmeans'
        self.train(X, n_clusters, **kwargs)
        self.mode = old_mode

    def evaluate(self, X: np.ndarray, sample_size: int = 1000) -> Dict:
        """Avalia o modelo de clustering.
        
        Args:
            X: Dados originais
            sample_size: Tamanho da amostra para cálculos
            
        Returns:
            Dict com métricas de avaliação
        """
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
                
                if self.mode == 'kmodes':
                    metrics['silhouette_score'] = float(silhouette_score(X_sample, sample_labels, metric='hamming'))
                else:
                    metrics['silhouette_score'] = float(silhouette_score(X_sample, sample_labels))
            else:
                if self.mode == 'kmodes':
                    metrics['silhouette_score'] = float(silhouette_score(X, labels, metric='hamming'))
                else:
                    metrics['silhouette_score'] = float(silhouette_score(X, labels))
            
            # Davies-Bouldin e Calinski-Harabasz requerem dados numéricos
            if self.mode == 'kmeans':
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
            else:
                metrics['davies_bouldin_score'] = float('nan')
                metrics['calinski_harabasz_score'] = float('nan')
        else:
            metrics['silhouette_score'] = float('nan')
            metrics['davies_bouldin_score'] = float('nan')
            metrics['calinski_harabasz_score'] = float('nan')
        
        # Adiciona custo/inércia
        if self.mode == 'kmodes' and hasattr(self.model, 'cost_'):
            metrics['cost'] = float(self.model.cost_)
        elif hasattr(self.model, 'inertia_'):
            metrics['inertia'] = float(self.model.inertia_)
        
        return metrics

    def reduce_dimensions(self, X: np.ndarray, n_components: int = 3) -> np.ndarray:
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        return self.pca.fit_transform(X)

    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """Salva o modelo treinado."""
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'mode': self.mode,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
        }
        joblib.dump(payload, filepath)
        logger.info(f"Modelo de clusterização ({self.mode}) salvo em: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Carrega um modelo salvo."""
        payload = joblib.load(filepath)
        self.model = payload['model']
        self.scaler = payload['scaler']
        self.pca = payload.get('pca')
        self.feature_names = payload.get('feature_names')
        self.random_state = payload.get('random_state', self.random_state)
        self.mode = payload.get('mode', 'kmeans')
        self.categorical_cols = payload.get('categorical_cols')
        self.numeric_cols = payload.get('numeric_cols')
        self.labels_ = getattr(self.model, 'labels_', None)
        logger.info(f"Modelo de clusterização ({self.mode}) carregado de: {filepath}")