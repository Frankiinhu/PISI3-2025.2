"""
Módulo de Clusterização para Identificação de Padrões
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import joblib
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseClusterer:
    """Classe para clusterização de padrões de doenças"""
    
    def __init__(self, random_state=42):
        """
        Inicializa o clusterizador
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.labels_ = None
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame, exclude_cols=None) -> np.ndarray:
        """
        Prepara dados para clusterização
        
        Args:
            df: DataFrame com dados
            exclude_cols: Colunas para excluir
            
        Returns:
            Array com dados escalados
        """
        if exclude_cols is None:
            exclude_cols = ['Diagnóstico']
            
        # Selecionar apenas features numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Excluir colunas especificadas
        for col in exclude_cols:
            if col in numeric_df.columns:
                numeric_df = numeric_df.drop(col, axis=1)
                
        self.feature_names = numeric_df.columns.tolist()
        
        # Escalar dados
        X_scaled = self.scaler.fit_transform(numeric_df)
        
        logger.info(f"Dados preparados para clusterização: {X_scaled.shape}")
        return X_scaled
        
    def find_optimal_clusters(self, X: np.ndarray, max_clusters=10) -> Dict:
        """
        Encontra número ótimo de clusters usando método do cotovelo
        
        Args:
            X: Dados para clusterização
            max_clusters: Número máximo de clusters a testar
            
        Returns:
            Dicionário com scores para cada número de clusters
        """
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            
        results = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
        
        # Encontrar melhor k baseado no silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"Melhor número de clusters (silhouette): {best_k}")
        
        return results
        
    def train_kmeans(self, X: np.ndarray, n_clusters=3, **kwargs) -> None:
        """
        Treina modelo K-Means
        
        Args:
            X: Dados para clusterização
            n_clusters: Número de clusters
            **kwargs: Parâmetros adicionais
        """
        params = {
            'n_clusters': n_clusters,
            'random_state': self.random_state,
            'n_init': kwargs.get('n_init', 10),
            'max_iter': kwargs.get('max_iter', 300)
        }
        
        self.model = KMeans(**params)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        
        logger.info(f"K-Means treinado com {n_clusters} clusters")
        
    def train_dbscan(self, X: np.ndarray, eps=0.5, min_samples=5) -> None:
        """
        Treina modelo DBSCAN
        
        Args:
            X: Dados para clusterização
            eps: Distância máxima entre pontos
            min_samples: Número mínimo de amostras
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        logger.info(f"DBSCAN: {n_clusters} clusters encontrados, {n_noise} pontos de ruído")
        
    def train_hierarchical(self, X: np.ndarray, n_clusters=3, linkage='ward') -> None:
        """
        Treina modelo de Clusterização Hierárquica
        
        Args:
            X: Dados para clusterização
            n_clusters: Número de clusters
            linkage: Tipo de linkage ('ward', 'complete', 'average')
        """
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        
        logger.info(f"Clusterização Hierárquica com {n_clusters} clusters")
        
    def evaluate(self, X: np.ndarray) -> Dict:
        """
        Avalia a qualidade da clusterização
        
        Args:
            X: Dados clusterizados
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if self.labels_ is None:
            raise ValueError("Modelo não treinado.")
            
        # Remover pontos de ruído para DBSCAN
        mask = self.labels_ != -1
        X_filtered = X[mask]
        labels_filtered = self.labels_[mask]
        
        if len(set(labels_filtered)) < 2:
            logger.warning("Menos de 2 clusters encontrados, métricas podem não ser confiáveis")
            
        metrics = {
            'n_clusters': len(set(self.labels_)) - (1 if -1 in self.labels_ else 0),
            'n_noise': list(self.labels_).count(-1),
            'silhouette_score': silhouette_score(X_filtered, labels_filtered) if len(set(labels_filtered)) > 1 else 0,
            'davies_bouldin_score': davies_bouldin_score(X_filtered, labels_filtered) if len(set(labels_filtered)) > 1 else 0,
            'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered) if len(set(labels_filtered)) > 1 else 0
        }
        
        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
        
        return metrics
        
    def reduce_dimensions(self, X: np.ndarray, n_components=2) -> np.ndarray:
        """
        Reduz dimensionalidade usando PCA para visualização
        
        Args:
            X: Dados para redução
            n_components: Número de componentes
            
        Returns:
            Array com dados reduzidos
        """
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_reduced = self.pca.fit_transform(X)
        
        explained_var = sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA: {explained_var:.2%} da variância explicada com {n_components} componentes")
        
        return X_reduced
        
    def get_cluster_profiles(self, df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        """
        Cria perfis de cada cluster
        
        Args:
            df: DataFrame original
            X_scaled: Dados escalados usados para clusterização
            
        Returns:
            DataFrame com perfis dos clusters
        """
        if self.labels_ is None:
            raise ValueError("Modelo não treinado.")
            
        # Adicionar labels ao DataFrame original
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = self.labels_
        
        # Calcular estatísticas por cluster
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cluster_profiles = df_with_clusters.groupby('Cluster')[numeric_cols].mean()
        
        return cluster_profiles
        
    def identify_risk_factors(self, df: pd.DataFrame, cluster_id: int) -> Dict:
        """
        Identifica fatores de risco para um cluster específico
        
        Args:
            df: DataFrame com dados e cluster labels
            cluster_id: ID do cluster para análise
            
        Returns:
            Dicionário com fatores de risco identificados
        """
        if self.labels_ is None:
            raise ValueError("Modelo não treinado.")
            
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = self.labels_
        
        # Dados do cluster específico
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        other_data = df_with_clusters[df_with_clusters['Cluster'] != cluster_id]
        
        # Comparar médias
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        risk_factors = {}
        for col in numeric_cols:
            cluster_mean = cluster_data[col].mean()
            other_mean = other_data[col].mean()
            diff = cluster_mean - other_mean
            
            if abs(diff) > 0.1:  # Threshold para considerar relevante
                risk_factors[col] = {
                    'cluster_mean': cluster_mean,
                    'other_mean': other_mean,
                    'difference': diff,
                    'relative_diff': (diff / other_mean * 100) if other_mean != 0 else 0
                }
                
        # Ordenar por diferença absoluta
        risk_factors = dict(sorted(risk_factors.items(), 
                                  key=lambda x: abs(x[1]['difference']), 
                                  reverse=True))
        
        return risk_factors
        
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz cluster para novos dados (apenas K-Means)
        
        Args:
            X: Novos dados
            
        Returns:
            Array com labels de cluster
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
            
        if not hasattr(self.model, 'predict'):
            raise ValueError("Método predict não disponível para este tipo de clusterização")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo de clusterização
        
        Args:
            filepath: Caminho para salvar
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'labels': self.labels_,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo de clusterização salvo em: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carrega um modelo salvo
        
        Args:
            filepath: Caminho do modelo
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.labels_ = model_data['labels']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Modelo de clusterização carregado de: {filepath}")
