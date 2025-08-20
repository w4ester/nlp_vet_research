import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VeterinaryUrgencyClassifier:
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        if model_type == 'ensemble':
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.models['gradient_boost'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.models['logistic'] = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
        else:
            self.models['main'] = self._get_model(model_type)
    
    def _get_model(self, model_type):
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )
        else:
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        feature_cols = []
        
        symptom_cols = [col for col in df.columns if col.startswith('symptom_')]
        feature_cols.extend(symptom_cols)
        
        text_feature_cols = [
            'text_length', 'exclamation_count', 'question_count',
            'critical_keyword_count', 'urgent_keyword_count',
            'moderate_keyword_count', 'low_keyword_count', 'urgency_words'
        ]
        for col in text_feature_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        numeric_cols = ['age_years', 'weight_kg', 'species_encoded', 'age_category_encoded']
        for col in numeric_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        X = df[feature_cols].fillna(0).values
        
        return X, feature_cols
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        print("Preparing training features...")
        X_train, feature_names = self.prepare_features(train_df)
        
        if 'urgency_score' in train_df.columns:
            y_train = train_df['urgency_score'].values
        elif 'urgency_numeric' in train_df.columns:
            y_train = train_df['urgency_numeric'].values
        else:
            raise ValueError("No target column found")
        
        y_train = np.clip(y_train, 1, 5).astype(int)
        
        print(f"Training with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            print(f"{model_name} CV F1 Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(
                    zip(feature_names, model.feature_importances_)
                )
        
        if val_df is not None:
            self.evaluate(val_df)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.prepare_features(df)
        
        if self.model_type == 'ensemble':
            predictions = []
            weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'logistic': 0.2}
            
            for model_name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred * weights.get(model_name, 1.0))
            
            final_predictions = np.round(np.mean(predictions, axis=0)).astype(int)
        else:
            final_predictions = self.models['main'].predict(X)
        
        return np.clip(final_predictions, 1, 5)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.prepare_features(df)
        
        if self.model_type == 'ensemble':
            probabilities = []
            weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'logistic': 0.2}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities.append(proba * weights.get(model_name, 1.0))
            
            if probabilities:
                return np.mean(probabilities, axis=0)
        else:
            if hasattr(self.models['main'], 'predict_proba'):
                return self.models['main'].predict_proba(X)
        
        return None
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        X_test, _ = self.prepare_features(test_df)
        
        if 'urgency_score' in test_df.columns:
            y_test = test_df['urgency_score'].values
        elif 'urgency_numeric' in test_df.columns:
            y_test = test_df['urgency_numeric'].values
        else:
            raise ValueError("No target column found")
        
        y_test = np.clip(y_test, 1, 5).astype(int)
        
        predictions = self.predict(test_df)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_weighted': f1_score(y_test, predictions, average='weighted'),
            'f1_macro': f1_score(y_test, predictions, average='macro')
        }
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 (weighted): {metrics['f1_weighted']:.3f}")
        print(f"F1 (macro): {metrics['f1_macro']:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        self.performance_metrics = metrics
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        importance_summary = {}
        
        for model_name, importance_dict in self.feature_importance.items():
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            importance_summary[model_name] = sorted_features
        
        return importance_summary
    
    def save_model(self, filepath: str):
        model_data = {
            'model_type': self.model_type,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f"{filepath}_{model_name}.pkl")
        
        print(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(f"{filepath}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        self.model_type = model_data['model_type']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance = model_data['feature_importance']
        
        self.models = {}
        if self.model_type == 'ensemble':
            for model_name in ['random_forest', 'gradient_boost', 'logistic']:
                self.models[model_name] = joblib.load(f"{filepath}_{model_name}.pkl")
        else:
            self.models['main'] = joblib.load(f"{filepath}_main.pkl")

def main():
    print("Loading processed datasets...")
    train_df = pd.read_csv('data/processed/train_processed.csv')
    val_df = pd.read_csv('data/processed/val_processed.csv')
    test_df = pd.read_csv('data/processed/test_processed.csv')
    
    print("\nInitializing urgency classifier...")
    classifier = VeterinaryUrgencyClassifier(model_type='ensemble')
    
    print("\nTraining models...")
    classifier.train(train_df, val_df)
    
    print("\n" + "="*60)
    print("TESTING ON HELD-OUT TEST SET")
    print("="*60)
    metrics = classifier.evaluate(test_df)
    
    print("\n" + "="*60)
    print("TOP IMPORTANT FEATURES")
    print("="*60)
    importance = classifier.get_feature_importance(top_n=15)
    for model_name, features in importance.items():
        print(f"\n{model_name.upper()}:")
        for feat_name, score in features[:10]:
            print(f"  {feat_name}: {score:.4f}")
    
    print("\nSaving trained models...")
    classifier.save_model('data/models/urgency_classifier')
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    raw_test = pd.read_csv('data/raw/test_cases.csv')
    sample_indices = [0, 10, 20, 30, 40]
    
    for idx in sample_indices:
        sample = test_df.iloc[[idx]]
        raw_sample = raw_test.iloc[idx]
        
        prediction = classifier.predict(sample)[0]
        actual = sample['urgency_numeric'].values[0] if 'urgency_numeric' in sample.columns else sample['urgency_score'].values[0]
        
        print(f"\nCase {idx}:")
        print(f"  Species: {raw_sample['species']}")
        print(f"  Description: {raw_sample['description'][:80]}...")
        print(f"  Predicted Urgency: {prediction}")
        print(f"  Actual Urgency: {int(actual)}")
        print(f"  Match: {'✓' if prediction == int(actual) else '✗'}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()