# Configuração do Projeto
# Copie este arquivo para config.py e ajuste conforme necessário

import os

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Dataset
DATASET_FILENAME = 'DATASET FINAL WRDP.csv'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

# Modelos
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, 'classifier_model.pkl')
CLUSTERER_MODEL_PATH = os.path.join(MODELS_DIR, 'clustering_model.pkl')

# Parâmetros de treinamento
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Classificador
CLASSIFIER_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

# Clusterização
MAX_CLUSTERS = 10

# API
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# Dashboard
DASHBOARD_HOST = '127.0.0.1'
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# Cores do Dashboard
DASHBOARD_COLORS = {
    'background': '#0f1419',
    'primary': '#1DA1F2',
    'secondary': '#14171A',
    'text': '#E1E8ED',
    'accent': '#00C9A7'
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
