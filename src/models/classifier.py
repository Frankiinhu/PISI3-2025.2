"""
Módulo de Classificação para Diagnóstico
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score)
import joblib
import logging
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosisClassifier:
    """Classe para classificação de diagnósticos"""
    
    def __init__(self, random_state=42):
        """
        Inicializa o classificador
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importances = None
        
    def prepare_data(self, df: pd.DataFrame, target_col='Diagnóstico', 
                    test_size=0.2) -> Tuple:
        """
        Prepara dados para treinamento
        
        Args:
            df: DataFrame com dados
            target_col: Nome da coluna alvo
            test_size: Proporção de dados para teste
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separar features e target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Salvar nomes das features
        self.feature_names = X.columns.tolist()
        
        # Codificar target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        # Escalar features (opcional, mas recomendado para alguns modelos)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Dados preparados: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def train_random_forest(self, X_train, y_train, **kwargs) -> None:
        """
        Treina modelo Random Forest
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            **kwargs: Parâmetros adicionais para RandomForestClassifier
        """
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'random_state': self.random_state
        }
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # Calcular importância das features
        self.feature_importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        logger.info("Modelo Random Forest treinado com sucesso")
        
    def train_gradient_boosting(self, X_train, y_train, **kwargs) -> None:
        """
        Treina modelo Gradient Boosting
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            **kwargs: Parâmetros adicionais
        """
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 3),
            'random_state': self.random_state
        }
        
        self.model = GradientBoostingClassifier(**params)
        self.model.fit(X_train, y_train)
        
        self.feature_importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        logger.info("Modelo Gradient Boosting treinado com sucesso")
        
    def evaluate(self, X_test, y_test) -> Dict:
        """
        Avalia o modelo
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute um método de treinamento primeiro.")
            
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
        }
        
        logger.info(f"Acurácia: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
        
    def cross_validate(self, X, y, cv=5) -> Dict:
        """
        Realiza validação cruzada
        
        Args:
            X: Features
            y: Target
            cv: Número de folds
            
        Returns:
            Dicionário com scores de validação cruzada
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
            
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
        
        logger.info(f"Cross-validation Accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
        
    def predict(self, X) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Features para predição
            
        Returns:
            Array com predições
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        return self.label_encoder.inverse_transform(y_pred)
        
    def predict_proba(self, X) -> np.ndarray:
        """
        Retorna probabilidades de predição
        
        Args:
            X: Features para predição
            
        Returns:
            Array com probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def get_feature_importance(self, top_n=15) -> pd.Series:
        """
        Retorna importância das features
        
        Args:
            top_n: Número de top features
            
        Returns:
            Series com importâncias
        """
        if self.feature_importances is None:
            raise ValueError("Importância de features não disponível")
            
        return self.feature_importances.head(top_n)
        
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo treinado
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
            
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carrega um modelo salvo
        
        Args:
            filepath: Caminho do modelo salvo
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importances = model_data['feature_importances']
        
        logger.info(f"Modelo carregado de: {filepath}")
