"""
Módulo de carregamento e limpeza de dados
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Classe para carregar e limpar o dataset"""
    
    def __init__(self, filepath: str):
        """
        Inicializa o DataLoader
        
        Args:
            filepath: Caminho para o arquivo CSV
        """
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV
        
        Returns:
            DataFrame com os dados carregados
        """
        try:
            self.df = pd.read_csv(self.filepath)
            logger.info(f"Dados carregados com sucesso: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")
            return self.df
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
            
    def unify_pain_behind_eye_columns(self) -> pd.DataFrame:
        """
        Unifica as colunas 'pain_behind_eye' e 'pain_behind_the_eye'
        
        Returns:
            DataFrame com colunas unificadas
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        if 'pain_behind_eye' in self.df.columns and 'pain_behind_the_eye' in self.df.columns:
            self.df['pain_behind_eye_unified'] = self.df[['pain_behind_eye', 'pain_behind_the_eye']].max(axis=1)
            self.df = self.df.drop(columns=['pain_behind_eye', 'pain_behind_the_eye'])
            logger.info("Colunas 'pain_behind_eye' unificadas com sucesso")
        else:
            logger.warning("Colunas 'pain_behind_eye' não encontradas")
            
        return self.df
        
    def handle_missing_values(self, strategy='mean') -> pd.DataFrame:
        """
        Trata valores faltantes no dataset
        
        Args:
            strategy: Estratégia para preencher valores faltantes ('mean', 'median', 'mode', 'drop')
            
        Returns:
            DataFrame com valores faltantes tratados
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy in ['mean', 'median', 'mode']:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.df[col].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Valores faltantes tratados: {missing_before} -> {missing_after}")
        
        return self.df
        
    def get_feature_names(self) -> dict:
        """
        Retorna nomes de features categorizados
        
        Returns:
            Dicionário com features categorizadas
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            
        climatic_vars = ['Temperatura (°C)', 'Umidade', 'Velocidade do Vento (km/h)']
        demographic_vars = ['Idade', 'Gênero']
        target_var = 'Diagnóstico'
        
        symptom_vars = [col for col in self.df.columns 
                       if col not in climatic_vars + demographic_vars + [target_var]]
        
        return {
            'climatic': climatic_vars,
            'demographic': demographic_vars,
            'symptoms': symptom_vars,
            'target': target_var
        }
        
    def get_clean_data(self) -> pd.DataFrame:
        """
        Pipeline completo de limpeza de dados
        
        Returns:
            DataFrame limpo e pronto para análise
        """
        self.load_data()
        self.unify_pain_behind_eye_columns()
        self.handle_missing_values(strategy='mean')
        
        logger.info("Pipeline de limpeza concluído com sucesso")
        return self.df
