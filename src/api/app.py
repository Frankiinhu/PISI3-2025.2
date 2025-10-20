"""
API REST para Predição de Diagnóstico
Integração com aplicativos externos
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
import logging

# Adicionar src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.classifier import DiagnosisClassifier
from src.models.clustering import DiseaseClusterer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS para integração com apps

# Carregar modelos (ajuste os caminhos conforme necessário)
classifier = DiagnosisClassifier()
clusterer = DiseaseClusterer()

try:
    classifier.load_model('../../models/saved_models/classifier_model.pkl')
    logger.info("Modelo de classificação carregado com sucesso")
except Exception as e:
    logger.warning(f"Modelo de classificação não carregado: {e}")

try:
    clusterer.load_model('../../models/saved_models/clustering_model.pkl')
    logger.info("Modelo de clusterização carregado com sucesso")
except Exception as e:
    logger.warning(f"Modelo de clusterização não carregado: {e}")


@app.route('/')
def home():
    """Endpoint raiz com informações da API"""
    return jsonify({
        'api': 'VitaNimbus - Weather Related Disease Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Predição de diagnóstico',
            '/predict_batch': 'POST - Predição em lote',
            '/cluster': 'POST - Identificação de cluster',
            '/risk_factors': 'POST - Identificação de fatores de risco',
            '/health': 'GET - Status da API'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Verifica status da API"""
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier.model is not None,
        'clusterer_loaded': clusterer.model is not None
    })


@app.route('/predict', methods=['POST'])
def predict_diagnosis():
    """
    Predição de diagnóstico para um paciente
    
    Exemplo de payload:
    {
        "idade": 35,
        "temperatura": 38.5,
        "umidade": 75,
        "velocidade_vento": 15,
        "sintomas": {
            "Febre": 1,
            "Tosse": 1,
            "Fadiga": 1,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Construir features
        features = {
            'Idade': data.get('idade', 0),
            'Temperatura (°C)': data.get('temperatura', 0),
            'Umidade': data.get('umidade', 0),
            'Velocidade do Vento (km/h)': data.get('velocidade_vento', 0)
        }
        
        # Adicionar sintomas
        if 'sintomas' in data:
            features.update(data['sintomas'])
        
        # Converter para DataFrame
        X = pd.DataFrame([features])
        
        # Garantir que todas as features necessárias estão presentes
        if classifier.feature_names:
            for feature in classifier.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            X = X[classifier.feature_names]
        
        # Fazer predição
        prediction = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]
        
        # Preparar resposta
        response = {
            'diagnostico_predito': prediction,
            'probabilidades': {
                classe: float(prob) 
                for classe, prob in zip(classifier.label_encoder.classes_, probabilities)
            },
            'confianca': float(max(probabilities))
        }
        
        logger.info(f"Predição realizada: {prediction}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predição em lote para múltiplos pacientes
    
    Exemplo de payload:
    {
        "patients": [
            {"idade": 35, "temperatura": 38.5, ...},
            {"idade": 42, "temperatura": 37.2, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
            
        predictions = []
        
        for patient_data in data['patients']:
            # Construir features
            features = {
                'Idade': patient_data.get('idade', 0),
                'Temperatura (°C)': patient_data.get('temperatura', 0),
                'Umidade': patient_data.get('umidade', 0),
                'Velocidade do Vento (km/h)': patient_data.get('velocidade_vento', 0)
            }
            
            if 'sintomas' in patient_data:
                features.update(patient_data['sintomas'])
            
            X = pd.DataFrame([features])
            
            if classifier.feature_names:
                for feature in classifier.feature_names:
                    if feature not in X.columns:
                        X[feature] = 0
                X = X[classifier.feature_names]
            
            prediction = classifier.predict(X)[0]
            probabilities = classifier.predict_proba(X)[0]
            
            predictions.append({
                'diagnostico_predito': prediction,
                'confianca': float(max(probabilities))
            })
        
        logger.info(f"Predição em lote realizada: {len(predictions)} pacientes")
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/cluster', methods=['POST'])
def identify_cluster():
    """
    Identifica cluster para um paciente
    
    Exemplo de payload: mesmo formato de /predict
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Construir features (similar ao predict)
        features = {
            'Idade': data.get('idade', 0),
            'Temperatura (°C)': data.get('temperatura', 0),
            'Umidade': data.get('umidade', 0),
            'Velocidade do Vento (km/h)': data.get('velocidade_vento', 0)
        }
        
        if 'sintomas' in data:
            features.update(data['sintomas'])
        
        X = pd.DataFrame([features])
        
        # Predizer cluster
        cluster = clusterer.predict_cluster(X.values)[0]
        
        response = {
            'cluster_id': int(cluster),
            'descricao': f'Cluster {cluster} - Grupo de risco identificado'
        }
        
        logger.info(f"Cluster identificado: {cluster}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na identificação de cluster: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/risk_factors', methods=['POST'])
def get_risk_factors():
    """
    Retorna fatores de risco baseado nas características do paciente
    
    Exemplo de payload: mesmo formato de /predict
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Identificar fatores de risco baseado nos valores
        risk_factors = []
        
        # Temperatura
        temp = data.get('temperatura', 0)
        if temp > 38:
            risk_factors.append({
                'fator': 'Temperatura elevada',
                'valor': temp,
                'nivel_risco': 'alto' if temp > 39 else 'moderado'
            })
        
        # Idade
        idade = data.get('idade', 0)
        if idade > 60 or idade < 5:
            risk_factors.append({
                'fator': 'Faixa etária de risco',
                'valor': idade,
                'nivel_risco': 'moderado'
            })
        
        # Sintomas
        if 'sintomas' in data:
            sintomas_presentes = [s for s, v in data['sintomas'].items() if v == 1]
            if len(sintomas_presentes) > 5:
                risk_factors.append({
                    'fator': 'Múltiplos sintomas',
                    'valor': len(sintomas_presentes),
                    'nivel_risco': 'alto'
                })
        
        response = {
            'fatores_risco': risk_factors,
            'total_fatores': len(risk_factors),
            'avaliacao_geral': 'Alto risco' if len(risk_factors) > 2 else 'Risco moderado' if len(risk_factors) > 0 else 'Baixo risco'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro na análise de fatores de risco: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Retorna informações sobre os modelos carregados"""
    info = {
        'classifier': {
            'loaded': classifier.model is not None,
            'features': classifier.feature_names if classifier.feature_names else [],
            'classes': list(classifier.label_encoder.classes_) if classifier.label_encoder else []
        },
        'clusterer': {
            'loaded': clusterer.model is not None,
            'n_clusters': len(set(clusterer.labels_)) if clusterer.labels_ is not None else 0
        }
    }
    
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
