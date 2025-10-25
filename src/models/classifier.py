"""
Módulo de Classificação para Diagnóstico
"""
import logging
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class DiagnosisClassifier:
    """Classe para classificação de diagnósticos"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importances = None
        self.metrics: Dict[str, float] | None = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Diagnóstico',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Separar features e target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Salvar nomes das features
        self.feature_names = X.columns.tolist()

        # Codificar target
        y_encoded = self.label_encoder.fit_transform(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        # Escalar (alguns modelos se beneficiam; para árvores não é problema)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Dados preparados: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        params = dict(
            n_estimators=200,
            max_depth=None,
            random_state=self.random_state,
            n_jobs=-1
        )
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        params = dict(random_state=self.random_state)
        params.update(kwargs)
        self.model = GradientBoostingClassifier(**params)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }
        self.metrics = metrics
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        # Usa o modelo atual se existir; caso contrário, RandomForest padrão
        estimator = self.model if self.model is not None else RandomForestClassifier(
            n_estimators=200, random_state=self.random_state, n_jobs=-1
        )
        scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return {'mean_score': float(scores.mean()), 'std_score': float(scores.std())}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("O modelo atual não suporta predict_proba.")

    def get_feature_importance(self, top_n: int = 15) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        if hasattr(self.model, "feature_importances_") and self.feature_names is not None:
            importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
            return importances.sort_values(ascending=False).head(top_n)
        # Fallback: coef_ linear (ex.: LogisticRegression)
        if hasattr(self.model, "coef_") and self.feature_names is not None:
            coefs = np.abs(self.model.coef_).mean(axis=0)
            importances = pd.Series(coefs, index=self.feature_names)
            return importances.sort_values(ascending=False).head(top_n)
        raise AttributeError("Importância de features não disponível para este modelo.")

    def compare_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        candidates = {
            'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=None),
            'SVC (RBF)': SVC(kernel='rbf', probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
        }
        rows = []
        for name, clf in candidates.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            rows.append({
                'Modelo': name,
                'Acurácia': float(accuracy_score(y_test, y_pred)),
                'Precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'Recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'F1-Score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            })
        return pd.DataFrame(rows).sort_values('F1-Score', ascending=False)

    def save_model(self, filepath: str) -> None:
        payload = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'random_state': self.random_state,
        }
        joblib.dump(payload, filepath)
        logger.info(f"Modelo salvo em: {filepath}")

    def load_model(self, filepath: str) -> None:
        payload = joblib.load(filepath)
        self.model = payload['model']
        self.label_encoder = payload['label_encoder']
        self.scaler = payload['scaler']
        self.feature_names = payload['feature_names']
        self.metrics = payload.get('metrics')
        self.random_state = payload.get('random_state', self.random_state)
        logger.info(f"Modelo carregado de: {filepath}")