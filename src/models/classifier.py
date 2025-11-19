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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

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
        self.shap_values = None
        self.shap_data = None
        self.shap_explainer = None  # Armazena o explainer SHAP
        self.best_params = None  # Armazena melhores parâmetros do tuning
        self.smote_comparison_results = None  # Armazena comparação Base vs SMOTE

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
            'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        }
        self.metrics = metrics
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        # Usa o modelo atual se existir; caso contrário, RandomForest padrão
        estimator = self.model if self.model is not None else RandomForestClassifier(
            n_estimators=200, random_state=self.random_state, n_jobs=-1
        )
        # Use n_jobs=1 to avoid memory issues with parallel processing
        scores = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy', n_jobs=1)
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
        # Fallback: coef_ linear (ex.: LogisticRegression, SVC linear)
        if hasattr(self.model, "coef_") and self.feature_names is not None:
            coef_attr = getattr(self.model, "coef_", None)
            if coef_attr is not None:
                coefs = np.abs(coef_attr).mean(axis=0)
                importances = pd.Series(coefs, index=self.feature_names)
                return importances.sort_values(ascending=False).head(top_n)
        raise AttributeError("Importância de features não disponível para este modelo.")

    def compare_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        use_smote: bool = False
    ) -> pd.DataFrame:
        candidates = {
            'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=None),
            'SVC (RBF)': SVC(kernel='rbf', probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=300, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
        }
        
        # Apply SMOTE if requested
        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            X_train = np.asarray(X_train_resampled)
            y_train = np.asarray(y_train_resampled)
        
        rows = []
        for name, clf in candidates.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            rows.append({
                'Modelo': name,
                'Acurácia': float(accuracy_score(y_test, y_pred)),
                'Acurácia Balanceada': float(balanced_accuracy_score(y_test, y_pred)),
                'Precision (Macro)': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                'Recall (Macro)': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
                'F1-Score (Macro)': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            })
        return pd.DataFrame(rows).sort_values('Acurácia Balanceada', ascending=False)

    def train_with_smote(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        model_type: str = 'random_forest',
        **kwargs
    ) -> None:
        """
        Treina modelo aplicando SMOTE para balancear as classes.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            model_type: Tipo de modelo ('random_forest', 'gradient_boosting', etc.)
            **kwargs: Parâmetros adicionais para o modelo
        """
        logger.info("Aplicando SMOTE para balanceamento de classes...")
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train_smote = np.asarray(X_train_resampled)
        y_train_smote = np.asarray(y_train_resampled)
        
        logger.info(f"Classes antes do SMOTE: {np.bincount(y_train.astype(int))}")
        logger.info(f"Classes após SMOTE: {np.bincount(y_train_smote.astype(int))}")
        
        if model_type == 'random_forest':
            self.train_random_forest(X_train_smote, y_train_smote, **kwargs)
        elif model_type == 'gradient_boosting':
            self.train_gradient_boosting(X_train_smote, y_train_smote, **kwargs)
        else:
            raise ValueError(f"model_type '{model_type}' não suportado.")

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'random_forest',
        search_type: str = 'random',
        cv: int = 5,
        n_iter: int = 20,
        n_jobs: int = -1,
        verbose: int = 2
    ) -> Dict[str, Any]:
        """
        Realiza tuning de hiperparâmetros usando GridSearchCV ou RandomizedSearchCV.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            model_type: 'random_forest' ou 'gradient_boosting'
            search_type: 'grid' para GridSearchCV ou 'random' para RandomizedSearchCV
            cv: Número de folds para validação cruzada
            n_iter: Número de iterações para RandomizedSearchCV
            n_jobs: Número de jobs paralelos (-1 usa todos os cores)
            verbose: Nível de verbosidade (0, 1, 2)
            
        Returns:
            Dict com melhores parâmetros e score
        """
        logger.info("🔍 Iniciando tuning de hiperparâmetros...")
        logger.info(f"   Método: {search_type.upper()}SearchCV")
        logger.info(f"   Modelo: {model_type}")
        logger.info(f"   Validação Cruzada: {cv}-fold")
        
        # Definir espaços de busca para cada modelo
        if model_type == 'random_forest':
            param_distributions = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            base_estimator = RandomForestClassifier(random_state=self.random_state, n_jobs=n_jobs)
            
        elif model_type == 'gradient_boosting':
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
            base_estimator = GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"model_type '{model_type}' não suportado. Use 'random_forest' ou 'gradient_boosting'.")
        
        # Escolher tipo de busca
        if search_type == 'grid':
            logger.info(f"   Combinações possíveis: ~{np.prod([len(v) for v in param_distributions.values()])}")
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_distributions,
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
        elif search_type == 'random':
            logger.info(f"   Iterações: {n_iter}")
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            raise ValueError(f"search_type '{search_type}' inválido. Use 'grid' ou 'random'.")
        
        # Executar busca
        logger.info("⏳ Iniciando busca (isso pode demorar alguns minutos)...")
        search.fit(X_train, y_train)
        
        # Armazenar resultados
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        
        logger.info("✅ Tuning concluído!")
        logger.info(f"   Best CV Score: {search.best_score_:.4f}")
        logger.info(f"   Melhores parâmetros:")
        for param, value in search.best_params_.items():
            logger.info(f"      {param}: {value}")
        
        return {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }

    def calculate_shap_values(self, X_sample: np.ndarray, max_samples: int = 100) -> None:
        """
        Calcula SHAP values para explicabilidade do modelo.
        Suporta feature importance, beeswarm plots, e force plots.
        
        Args:
            X_sample: Amostra de dados para calcular SHAP values (já escalonada)
            max_samples: Número máximo de amostras (padrão: 100 para performance)
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute train_*() primeiro.")
        
        try:
            import shap
            
            # Limitar amostras para performance
            if len(X_sample) > max_samples:
                logger.info(f"📊 Amostrando {max_samples} de {len(X_sample)} registros para SHAP...")
                indices = np.random.choice(len(X_sample), max_samples, replace=False)
                X_sample_reduced = X_sample[indices]
            else:
                X_sample_reduced = X_sample
                logger.info(f"📊 Usando todas as {len(X_sample)} amostras para SHAP...")
            
            # Verificar se é modelo baseado em árvores
            if hasattr(self.model, 'estimators_'):
                logger.info("🌳 Usando TreeExplainer (otimizado para Random Forest/Gradient Boosting)...")
                explainer = shap.TreeExplainer(self.model)
                self.shap_explainer = explainer
                
                # Calcular SHAP values
                logger.info("⏳ Calculando SHAP values (pode demorar um pouco)...")
                self.shap_values = explainer.shap_values(X_sample_reduced)
                self.shap_data = X_sample_reduced
                
                # Verificar formato dos SHAP values
                if isinstance(self.shap_values, list):
                    n_classes = len(self.shap_values)
                    n_samples = self.shap_values[0].shape[0]
                    n_features = self.shap_values[0].shape[1]
                    logger.info(f"✅ SHAP values calculados:")
                    logger.info(f"   - {n_classes} classes")
                    logger.info(f"   - {n_samples} amostras")
                    logger.info(f"   - {n_features} features")
                elif isinstance(self.shap_values, np.ndarray):
                    logger.info(f"✅ SHAP values calculados: shape {self.shap_values.shape}")
                
                logger.info("📈 Pronto para visualizações:")
                logger.info("   ✓ Feature Importance (global)")
                logger.info("   ✓ Beeswarm Plot (global)")
                logger.info("   ✓ Bar Plot Multiclasse (global)")
                logger.info("   ✓ Force Plot (local - individual)")
                
            else:
                logger.warning("⚠️ Modelo não suporta TreeExplainer.")
                logger.warning("   Use Random Forest ou Gradient Boosting para SHAP completo.")
                self.shap_values = None
                self.shap_data = None
                self.shap_explainer = None
                
        except ImportError:
            logger.error("❌ Biblioteca 'shap' não encontrada!")
            logger.info("📦 Instale com: pip install shap")
            self.shap_values = None
            self.shap_data = None
            self.shap_explainer = None
        except Exception as e:
            logger.error(f"❌ Erro ao calcular SHAP values: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.shap_values = None
            self.shap_data = None
            self.shap_explainer = None
    
    def get_shap_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Retorna feature importance baseado em SHAP values (média absoluta).
        
        Args:
            top_n: Número de top features para retornar
            
        Returns:
            DataFrame com features e importâncias
        """
        if self.shap_values is None or self.feature_names is None:
            raise RuntimeError("SHAP values não calculados. Execute calculate_shap_values() primeiro.")
        
        # Para multiclasse, pegar média entre classes
        if isinstance(self.shap_values, list):
            # Lista de arrays (uma por classe)
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            # Array único ou 3D
            if len(self.shap_values.shape) == 3:
                # (samples, features, classes)
                mean_abs_shap = np.abs(self.shap_values).mean(axis=(0, 2))
            else:
                # (samples, features)
                mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df

    def compare_with_without_smote(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """
        Compara performance de modelos com e sem SMOTE.
        
        Returns:
            Dict com 'base' e 'smote' DataFrames de comparação
        """
        logger.info("Comparando modelos: Base vs SMOTE")
        
        # Modelos sem SMOTE
        df_base = self.compare_models(X_train, X_test, y_train, y_test, use_smote=False)
        df_base['Versão'] = 'Base'
        
        # Modelos com SMOTE
        df_smote = self.compare_models(X_train, X_test, y_train, y_test, use_smote=True)
        df_smote['Versão'] = 'Com SMOTE'
        
        # Store results for dashboard use
        self.smote_comparison_results = {'base': df_base, 'smote': df_smote}
        
        return self.smote_comparison_results

    def save_model(self, filepath: str) -> None:
        payload = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'random_state': self.random_state,
            'shap_values': self.shap_values,
            'shap_data': self.shap_data,
            'shap_explainer': None,  # Explainer não é serializável, recalcular se necessário
            'best_params': self.best_params,
            'smote_comparison_results': self.smote_comparison_results,
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
        self.shap_values = payload.get('shap_values')
        self.shap_data = payload.get('shap_data')
        self.shap_explainer = None  # Será recriado se necessário
        self.best_params = payload.get('best_params')
        self.smote_comparison_results = payload.get('smote_comparison_results')
        logger.info(f"Modelo carregado de: {filepath}")

    def explain_prediction(self, X_instance: np.ndarray, top_n: int = 5) -> list:
        """
        Explica uma predição individual usando SHAP values.
        
        Args:
            X_instance: Instância única para explicar (shape: 1, n_features)
            top_n: Número de features mais importantes para retornar
            
        Returns:
            Lista de dicionários com feature, impact (SHAP value) e value (input value)
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado.")
        
        try:
            import shap
        except ImportError:
            raise ImportError("Biblioteca 'shap' não instalada. Execute: pip install shap")
        
        # Criar explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_instance)
        
        # Fazer predição para saber qual classe foi predita
        prediction_idx = self.model.predict(X_instance)[0]
        
        # Obter SHAP values da classe predita
        if isinstance(shap_values, list):
            # Multiclass: lista de arrays (um por classe)
            class_shap_values = shap_values[prediction_idx][0]
        else:
            # Binary ou formato diferente
            class_shap_values = shap_values[0]
        
        explanations = []
        feature_names_list = self.feature_names if self.feature_names else [
            f"Feature {i}" for i in range(len(class_shap_values))
        ]
        
        for name, shap_val, input_val in zip(feature_names_list, class_shap_values, X_instance[0]):
            # Filtrar features com impacto muito baixo
            if abs(shap_val) > 0.01:
                explanations.append({
                    "feature": name,
                    "impact": float(shap_val),     # Valor SHAP (+ aumenta risco, - diminui)
                    "value": float(input_val)      # O valor que o usuário inseriu
                })
        
        # Ordenar por impacto absoluto (mais importante primeiro)
        explanations.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return explanations[:top_n]