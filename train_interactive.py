"""Script simplificado para treinar modelo com tuning e SHAP"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from scripts.train_models import train_classifier, train_clusterer

if __name__ == '__main__':
    # Caminhos
    data_path = Path('data/DATASET FINAL WRDP.csv')
    out_dir = Path('models/saved_models')
    
    if not data_path.exists():
        print(f"ERRO: Dataset não encontrado em {data_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("NimbusVita - Treinamento com Tuning de Hiperparâmetros")
    print("="*70)
    print("\nEscolha uma opção:")
    print("1. Treinamento Básico (rápido, ~5-10 min)")
    print("2. Com SMOTE (~10-15 min)")
    print("3. Com RandomSearch Tuning (~30-45 min)")
    print("4. Completo: SMOTE + RandomSearch + SHAP (~45-60 min)")
    print("5. Com GridSearch (muito lento, ~1-2 horas)")
    print()
    
    choice = input("Digite sua escolha (1-5): ").strip()
    
    use_smote = choice in ['2', '4']
    tune_hyperparams = choice in ['3', '4', '5']
    search_type = 'grid' if choice == '5' else 'random'
    
    print("\n" + "-"*70)
    print("CONFIGURAÇÃO:")
    print(f"  SMOTE: {use_smote}")
    print(f"  Tuning: {tune_hyperparams}")
    if tune_hyperparams:
        print(f"  Tipo de Busca: {search_type}")
    print("-"*70 + "\n")
    
    # Treinar classificador
    clf, metrics = train_classifier(
        str(data_path),
        str(out_dir / 'classifier.joblib'),
        use_smote=use_smote,
        tune_hyperparams=tune_hyperparams,
        search_type=search_type
    )
    
    # Treinar clusterizador
    print("\n" + "-"*70)
    clusterer, cluster_metrics = train_clusterer(
        str(data_path),
        str(out_dir / 'clusterer.joblib')
    )
    
    print("\n" + "="*70)
    print("TREINAMENTO CONCLUIDO COM SUCESSO!")
    print("="*70)
    print("\nPróxima etapa: Execute o dashboard")
    print("  python dashboard/app_complete.py")
    print("\nAcesse: http://127.0.0.1:8050/")
    print("="*70 + "\n")
