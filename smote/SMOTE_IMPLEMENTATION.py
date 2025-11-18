import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import imblearn

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Imbalanced-learn (SMOTE)
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import classification_report_imbalanced

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.data_loader import DataLoader


class SMOTEAnalyzer:
    """Analyzer to compare model performance with and without SMOTE"""
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize the SMOTE Analyzer
        
        Args:
            data_path: Path to the dataset CSV file
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Results storage
        self.results = {
            'baseline': {},
            'smote': {},
            'smote_nc': {},
            'borderline_smote': {}
        }
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the dataset"""
        loader = DataLoader(self.data_path)
        df = loader.get_clean_data()
        
        X = df.drop(columns=['Diagnóstico'])
        y = df['Diagnóstico']
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"\nClass distribution:")
        print(y.value_counts())
        
        return X, y
    
    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets"""
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTrain set: {len(X_train_scaled)} samples")
        print(f"Test set: {len(X_test_scaled)} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_baseline_model(
        self, X_train: np.ndarray, X_test: np.ndarray, 
        y_train: np.ndarray, y_test: np.ndarray
    ) -> Dict:
        """Train baseline model WITHOUT SMOTE"""
        print("\n" + "="*70)
        print("TRAINING BASELINE MODEL (No SMOTE)")
        print("="*70)
        
        # Train Random Forest
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, "Baseline")
        self.results['baseline'] = results
        
        return results
    
    def train_smote_model(
        self, X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        method: str = 'smote'
    ) -> Dict:
        """Train model WITH SMOTE or variants"""
        print("\n" + "="*70)
        print(f"TRAINING MODEL WITH {method.upper()}")
        print("="*70)
        
        # Choose SMOTE variant
        if method == 'smote':
            resampler = SMOTE(random_state=self.random_state, k_neighbors=5)
        elif method == 'smote_nc':
            # SMOTE-NC for mixed data (requires categorical feature indices)
            # Assuming 'Gênero' (index 1) is categorical
            resampler = SMOTENC(
                categorical_features=[1],  # Gender column
                random_state=self.random_state,
                k_neighbors=5
            )
        elif method == 'borderline_smote':
            resampler = BorderlineSMOTE(
                random_state=self.random_state,
                k_neighbors=5
            )
        elif method == 'adasyn':
            resampler = ADASYN(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply SMOTE to training data
        print(f"Original training set size: {len(X_train)}")
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        print(f"Resampled training set size: {len(X_train_resampled)}")
        
        # Show new class distribution
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        print("\nClass distribution after SMOTE:")
        for cls, count in zip(unique, counts):
            disease_name = self.label_encoder.inverse_transform([cls])[0]
            print(f"  {disease_name}: {count}")
        
        # Train Random Forest on resampled data
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train_resampled, y_train_resampled)
        
        # Predictions on original test set
        y_pred = clf.predict(X_test)
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, method.upper())
        self.results[method] = results
        
        return results
    
    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, label: str
    ) -> Dict:
        """Calculate and print metrics"""
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{label} Model Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def compare_results(self):
        """Compare results between baseline and SMOTE models"""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        comparison_data = []
        for method, results in self.results.items():
            if results:
                comparison_data.append({
                    'Method': method.upper(),
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Precision': f"{results['precision']:.4f}",
                    'Recall': f"{results['recall']:.4f}",
                    'F1-Score': f"{results['f1_score']:.4f}"
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n", df_comparison.to_string(index=False))
        
        # Calculate improvements
        if 'baseline' in self.results and 'smote' in self.results:
            baseline = self.results['baseline']
            smote = self.results['smote']
            
            print("\n" + "="*70)
            print("IMPROVEMENTS WITH SMOTE")
            print("="*70)
            
            improvements = {
                'Accuracy': (smote['accuracy'] - baseline['accuracy']) * 100,
                'Precision': (smote['precision'] - baseline['precision']) * 100,
                'Recall': (smote['recall'] - baseline['recall']) * 100,
                'F1-Score': (smote['f1_score'] - baseline['f1_score']) * 100
            }
            
            for metric, improvement in improvements.items():
                symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"{symbol} {metric}: {improvement:+.2f}%")
    
    def plot_confusion_matrices(self, save_path: str = None):
        """Plot confusion matrices for comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        methods = ['baseline', 'smote']
        titles = ['Baseline (No SMOTE)', 'With SMOTE']
        
        for idx, (method, title) in enumerate(zip(methods, titles)):
            if method in self.results and self.results[method]:
                y_true = self.results[method]['y_true']
                y_pred = self.results[method]['y_pred']
                
                cm = confusion_matrix(y_true, y_pred)
                
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=axes[idx]
                )
                axes[idx].set_title(title, fontsize=14, fontweight='bold')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrices saved to: {save_path}")
        else:
            plt.show()
    
    def analyze_minority_class_performance(self):
        """Analyze performance specifically on minority classes"""
        print("\n" + "="*70)
        print("MINORITY CLASS PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Define minority classes (bottom 5 by count)
        minority_classes = ['Sinusitis', 'Arthritis', 'Common Cold', 'Dengue', 'Eczema']
        
        for method in ['baseline', 'smote']:
            if method not in self.results or not self.results[method]:
                continue
                
            print(f"\n{method.upper()} - Minority Class Metrics:")
            y_true = self.results[method]['y_true']
            y_pred = self.results[method]['y_pred']
            
            # Get per-class metrics
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=range(len(self.label_encoder.classes_)), zero_division=0
            )
            
            minority_indices = [
                i for i, name in enumerate(self.label_encoder.classes_) 
                if name in minority_classes
            ]
            
            avg_precision = np.mean([precision[i] for i in minority_indices])
            avg_recall = np.mean([recall[i] for i in minority_indices])
            avg_f1 = np.mean([f1[i] for i in minority_indices])
            
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall:    {avg_recall:.4f}")
            print(f"  Average F1-Score:  {avg_f1:.4f}")
    
    def run_full_analysis(self):
        """Run complete SMOTE analysis pipeline"""
        print("\n" + "🔍"*35)
        print("SMOTE ANALYSIS FOR PISI3 DATASET")
        print("🔍"*35)
        
        # Load data
        X, y = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train baseline
        self.train_baseline_model(X_train, X_test, y_train, y_test)
        
        # Train with SMOTE
        self.train_smote_model(X_train, X_test, y_train, y_test, method='smote')
        
        # Train with SMOTE-NC (handles categorical features better)
        try:
            self.train_smote_model(X_train, X_test, y_train, y_test, method='smote_nc')
        except Exception as e:
            print(f"\nSMOTE-NC failed: {e}")
        
        # Train with Borderline-SMOTE
        try:
            self.train_smote_model(X_train, X_test, y_train, y_test, method='borderline_smote')
        except Exception as e:
            print(f"\nBorderline-SMOTE failed: {e}")
        
        # Compare results
        self.compare_results()
        
        # Minority class analysis
        self.analyze_minority_class_performance()
        
        # Plot confusion matrices
        self.plot_confusion_matrices(save_path='smote_confusion_matrices.png')
        
        print("✅ ANALYSIS COMPLETE")


def main():
    """Main execution function"""
    # Path to dataset
    data_path = Path(__file__).parent / "data" / "DATASET FINAL WRDP.csv"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return
    
    # Run analysis
    analyzer = SMOTEAnalyzer(str(data_path), random_state=42)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
