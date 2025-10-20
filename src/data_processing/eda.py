"""
Módulo de Análise Exploratória de Dados (EDA)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExploratoryDataAnalysis:
    """Classe para realizar análise exploratória de dados"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa a classe EDA
        
        Args:
            df: DataFrame para análise
        """
        self.df = df
        self.stats = {}
        
    def basic_info(self) -> Dict:
        """
        Retorna informações básicas sobre o dataset
        
        Returns:
            Dicionário com informações básicas
        """
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        
        logger.info(f"Dataset: {info['shape'][0]} linhas, {info['shape'][1]} colunas")
        logger.info(f"Valores duplicados: {info['duplicates']}")
        
        return info
        
    def numerical_statistics(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Calcula estatísticas descritivas para colunas numéricas
        
        Args:
            columns: Lista de colunas para análise (None para todas)
            
        Returns:
            DataFrame com estatísticas descritivas
        """
        if columns is None:
            numeric_df = self.df.select_dtypes(include=[np.number])
        else:
            numeric_df = self.df[columns]
            
        stats = numeric_df.describe()
        
        # Adicionar informações extras
        stats.loc['skewness'] = numeric_df.skew()
        stats.loc['kurtosis'] = numeric_df.kurtosis()
        
        return stats
        
    def categorical_distribution(self, column: str) -> pd.DataFrame:
        """
        Analisa distribuição de uma variável categórica
        
        Args:
            column: Nome da coluna categórica
            
        Returns:
            DataFrame com contagens e percentuais
        """
        counts = self.df[column].value_counts()
        percentages = self.df[column].value_counts(normalize=True) * 100
        
        result = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })
        
        return result
        
    def symptom_frequency_by_diagnosis(self, symptom: str) -> pd.DataFrame:
        """
        Calcula frequência de um sintoma por diagnóstico
        
        Args:
            symptom: Nome do sintoma
            
        Returns:
            DataFrame com frequências
        """
        if symptom not in self.df.columns:
            raise ValueError(f"Sintoma '{symptom}' não encontrado no dataset")
            
        if 'Diagnóstico' not in self.df.columns:
            raise ValueError("Coluna 'Diagnóstico' não encontrada")
            
        frequency = self.df.groupby('Diagnóstico')[symptom].sum().reset_index()
        frequency.columns = ['Diagnóstico', 'Contagem']
        
        return frequency
        
    def correlation_matrix(self, columns: List[str] = None, method='pearson') -> pd.DataFrame:
        """
        Calcula matriz de correlação
        
        Args:
            columns: Lista de colunas (None para todas numéricas)
            method: Método de correlação ('pearson', 'spearman', 'kendall')
            
        Returns:
            DataFrame com matriz de correlação
        """
        if columns is None:
            numeric_df = self.df.select_dtypes(include=[np.number])
        else:
            numeric_df = self.df[columns]
            
        corr_matrix = numeric_df.corr(method=method)
        
        return corr_matrix
        
    def climate_vs_diagnosis_stats(self, climatic_vars: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Analisa estatísticas de variáveis climáticas por diagnóstico
        
        Args:
            climatic_vars: Lista de variáveis climáticas
            
        Returns:
            Dicionário com estatísticas por variável climática
        """
        if 'Diagnóstico' not in self.df.columns:
            raise ValueError("Coluna 'Diagnóstico' não encontrada")
            
        stats_dict = {}
        
        for var in climatic_vars:
            if var in self.df.columns:
                stats = self.df.groupby('Diagnóstico')[var].agg([
                    'mean', 'median', 'std', 'min', 'max'
                ]).round(2)
                stats_dict[var] = stats
            else:
                logger.warning(f"Variável '{var}' não encontrada")
                
        return stats_dict
        
    def detect_outliers(self, column: str, method='iqr') -> pd.DataFrame:
        """
        Detecta outliers em uma coluna
        
        Args:
            column: Nome da coluna
            method: Método de detecção ('iqr' ou 'zscore')
            
        Returns:
            DataFrame com outliers detectados
        """
        if column not in self.df.columns:
            raise ValueError(f"Coluna '{column}' não encontrada")
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            outliers = self.df[z_scores > 3]
            
        logger.info(f"Outliers detectados em '{column}': {len(outliers)}")
        return outliers
        
    def get_comprehensive_report(self) -> Dict:
        """
        Gera relatório completo da análise exploratória
        
        Returns:
            Dicionário com todas as análises
        """
        report = {
            'basic_info': self.basic_info(),
            'numerical_stats': self.numerical_statistics(),
            'diagnosis_distribution': self.categorical_distribution('Diagnóstico') if 'Diagnóstico' in self.df.columns else None,
        }
        
        logger.info("Relatório completo gerado com sucesso")
        return report
