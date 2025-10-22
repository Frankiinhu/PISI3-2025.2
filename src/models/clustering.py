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
from typing import Dict, Tuple, Optional, List

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
        self.feature_names: Optional[List[str]] = None
        
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

    def transform_with_training_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma dados usando as mesmas features e ordem do treinamento."""
        if self.feature_names is None:
            raise ValueError("Feature names não definidos; treine ou carregue o clusterizador antes de transformar dados.")

        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler não está ajustado; execute o treinamento antes de transformar dados.")

        numeric_df = df.select_dtypes(include=[np.number])
        missing = [col for col in self.feature_names if col not in numeric_df.columns]
        if missing:
            missing_str = ', '.join(missing)
            raise ValueError(f"Colunas ausentes para transformar dados com o clusterizador: {missing_str}")

        ordered_data = numeric_df[self.feature_names].copy()
        return self.scaler.transform(ordered_data)
        
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
        k_range = list(range(2, max_clusters + 1))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            
        # Encontrar melhor k baseado no silhouette score
        silhouette_best_k = k_range[int(np.argmax(silhouette_scores))]

        # Método do cotovelo (distância do ponto à linha entre extremidades)
        start_point = np.array([k_range[0], inertias[0]])
        end_point = np.array([k_range[-1], inertias[-1]])
        vector = end_point - start_point
        norm = np.linalg.norm(vector)
        elbow_distances = []

        if norm == 0:
            elbow_distances = [0.0 for _ in k_range]
            elbow_k = k_range[0]
        else:
            for k_value, inertia in zip(k_range, inertias):
                point = np.array([k_value, inertia])
                distance = np.abs(np.cross(vector, start_point - point)) / norm
                elbow_distances.append(float(distance))
            elbow_k = k_range[int(np.argmax(elbow_distances))]

        logger.info(f"Melhor número de clusters (silhouette): {silhouette_best_k}")
        logger.info(f"Sugestão pelo método do cotovelo: {elbow_k}")

        return {
            'k_values': k_range,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'silhouette_best_k': silhouette_best_k,
            'elbow_distances': elbow_distances,
            'elbow_k': elbow_k
        }
        
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
    
    def compare_clustering_methods(self, X: np.ndarray, n_clusters=3) -> pd.DataFrame:
        """
        Compara diferentes métodos de clusterização
        
        Args:
            X: Dados para clusterização
            n_clusters: Número de clusters (para métodos que exigem)
            
        Returns:
            DataFrame com métricas comparativas
        """
        import time
        
        methods = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10),
            'K-Means++': KMeans(n_clusters=n_clusters, init='k-means++', random_state=self.random_state, n_init=10),
            'DBSCAN (eps=0.5)': DBSCAN(eps=0.5, min_samples=5),
            'DBSCAN (eps=1.0)': DBSCAN(eps=1.0, min_samples=5),
            'Hierarchical (Ward)': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
            'Hierarchical (Complete)': AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
            'Hierarchical (Average)': AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        }
        
        results = []
        
        for name, model in methods.items():
            logger.info(f"Testando {name}...")
            start_time = time.time()
            
            try:
                # Treinar
                if hasattr(model, 'fit_predict'):
                    labels = model.fit_predict(X)
                else:
                    model.fit(X)
                    labels = model.labels_
                
                training_time = time.time() - start_time
                
                # Remover pontos de ruído para DBSCAN
                mask = labels != -1
                X_filtered = X[mask]
                labels_filtered = labels[mask]
                
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Calcular métricas (apenas se tiver pelo menos 2 clusters)
                if len(set(labels_filtered)) > 1:
                    silhouette = silhouette_score(X_filtered, labels_filtered)
                    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
                    calinski = calinski_harabasz_score(X_filtered, labels_filtered)
                else:
                    silhouette = davies_bouldin = calinski = np.nan
                
                results.append({
                    'Método': name,
                    'N_Clusters': n_clusters_found,
                    'Ruído': n_noise,
                    'Silhouette': silhouette,
                    'Davies-Bouldin': davies_bouldin,
                    'Calinski-Harabasz': calinski,
                    'Tempo (s)': training_time
                })
                
            except Exception as e:
                logger.warning(f"Erro ao treinar {name}: {e}")
                results.append({
                    'Método': name,
                    'N_Clusters': np.nan,
                    'Ruído': np.nan,
                    'Silhouette': np.nan,
                    'Davies-Bouldin': np.nan,
                    'Calinski-Harabasz': np.nan,
                    'Tempo (s)': np.nan
                })
        
        results_df = pd.DataFrame(results)
        
        # Ordenar por Silhouette Score (maior é melhor)
        results_df_sorted = results_df.sort_values('Silhouette', ascending=False, na_position='last')
        
        logger.info("\n=== COMPARAÇÃO DE MÉTODOS DE CLUSTERIZAÇÃO ===")
        logger.info(f"\n{results_df_sorted.to_string()}")
        
        return results_df_sorted
    
    def reduce_dimensions_tsne(self, X: np.ndarray, n_components=2, perplexity=30) -> np.ndarray:
        """
        Reduz dimensionalidade usando t-SNE para visualização
        (Melhor que PCA para visualização de clusters)
        
        Args:
            X: Dados para redução
            n_components: Número de componentes (2 ou 3)
            perplexity: Perplexidade do t-SNE (5-50 recomendado)
            
        Returns:
            Array com dados reduzidos
        """
        from sklearn.manifold import TSNE
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            n_iter=1000
        )
        
        X_reduced = tsne.fit_transform(X)
        
        logger.info(f"t-SNE: Redução para {n_components}D concluída")
        
        return X_reduced
    
    def calculate_gap_statistic(self, X: np.ndarray, max_clusters=10, n_refs=10) -> Dict:
        """
        Calcula Gap Statistic para determinar número ótimo de clusters
        (Método estatístico mais robusto que o Elbow)
        
        Args:
            X: Dados para clusterização
            max_clusters: Número máximo de clusters a testar
            n_refs: Número de datasets de referência
            
        Returns:
            Dicionário com Gap Statistics
        """
        gaps = []
        std_gaps = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            # Clusterização nos dados reais
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            real_dispersion = np.log(kmeans.inertia_)
            
            # Clusterização em dados de referência (uniformes)
            ref_dispersions = []
            for _ in range(n_refs):
                # Gerar dados de referência uniformes
                random_data = np.random.uniform(
                    X.min(axis=0),
                    X.max(axis=0),
                    size=X.shape
                )
                
                kmeans_ref = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans_ref.fit(random_data)
                ref_dispersions.append(np.log(kmeans_ref.inertia_))
            
            # Calcular Gap
            ref_dispersion_mean = np.mean(ref_dispersions)
            gap = ref_dispersion_mean - real_dispersion
            
            # Calcular desvio padrão
            sdk = np.std(ref_dispersions)
            sk = sdk * np.sqrt(1 + 1/n_refs)
            
            gaps.append(gap)
            std_gaps.append(sk)
        
        # Encontrar k ótimo (primeiro k onde gap(k) >= gap(k+1) - std(k+1))
        optimal_k = 1
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i+1] - std_gaps[i+1]:
                optimal_k = k_range[i]
                break
        
        logger.info(f"Gap Statistic - Número ótimo de clusters: {optimal_k}")
        
        return {
            'k_values': list(k_range),
            'gaps': gaps,
            'std_gaps': std_gaps,
            'optimal_k': optimal_k
        }
    
    def describe_cluster_clinically(self, df: pd.DataFrame, cluster_id: int) -> str:
        """
        Gera descrição clínica textual de um cluster
        
        Args:
            df: DataFrame original com dados
            cluster_id: ID do cluster para descrever
            
        Returns:
            String com descrição clínica do cluster
        """
        if self.labels_ is None:
            raise ValueError("Modelo não treinado.")
        
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = self.labels_
        
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        
        # Estatísticas do cluster
        n_patients = len(cluster_data)
        pct_total = (n_patients / len(df)) * 100
        
        # Diagnósticos mais comuns (se disponível)
        diagnosis_dist = ""
        if 'Diagnóstico' in cluster_data.columns:
            top_diagnosis = cluster_data['Diagnóstico'].value_counts().head(3)
            diagnosis_dist = "\n  Diagnósticos principais:\n"
            for diag, count in top_diagnosis.items():
                diagnosis_dist += f"    - {diag}: {count} casos ({count/n_patients*100:.1f}%)\n"
        
        # Variáveis numéricas principais
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        top_vars = []
        
        for col in numeric_cols:
            cluster_mean = cluster_data[col].mean()
            overall_mean = df[col].mean()
            diff_pct = ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
            
            if abs(diff_pct) > 20:  # Diferença significativa
                direction = "elevado" if diff_pct > 0 else "reduzido"
                top_vars.append(f"    - {col}: {direction} em {abs(diff_pct):.1f}% vs média geral")
        
        characteristics = "\n".join(top_vars[:5]) if top_vars else "    - Sem características distintivas"
        
        description = f"""
=== PERFIL CLÍNICO DO CLUSTER {cluster_id} ===

Tamanho: {n_patients} pacientes ({pct_total:.1f}% do total)
{diagnosis_dist}
Características distintivas:
{characteristics}

Interpretação: Este cluster representa um subgrupo de pacientes com perfil 
{"clínico diferenciado" if len(top_vars) > 0 else "similar à população geral"}.
"""
        
        return description
        
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