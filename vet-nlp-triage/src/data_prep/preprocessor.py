import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

class VeterinaryDataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.symptom_vocabulary = set()
        self.symptom_to_idx = {}
        self.idx_to_symptom = {}
        
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_symptom_features(self, symptoms: List[str]) -> Dict[str, int]:
        features = {}
        for symptom in symptoms:
            clean_symptom = self.clean_text(symptom)
            self.symptom_vocabulary.add(clean_symptom)
            features[clean_symptom] = 1
        return features
    
    def build_symptom_vocabulary(self, df: pd.DataFrame):
        all_symptoms = []
        for symptoms_list in df['symptoms']:
            if isinstance(symptoms_list, str):
                symptoms_list = eval(symptoms_list)
            for symptom in symptoms_list:
                clean_symptom = self.clean_text(symptom)
                all_symptoms.append(clean_symptom)
        
        unique_symptoms = sorted(set(all_symptoms))
        self.symptom_to_idx = {symptom: idx for idx, symptom in enumerate(unique_symptoms)}
        self.idx_to_symptom = {idx: symptom for symptom, idx in self.symptom_to_idx.items()}
        self.symptom_vocabulary = set(unique_symptoms)
    
    def encode_symptoms(self, symptoms: List[str]) -> np.ndarray:
        encoding = np.zeros(len(self.symptom_to_idx))
        if isinstance(symptoms, str):
            symptoms = eval(symptoms)
        for symptom in symptoms:
            clean_symptom = self.clean_text(symptom)
            if clean_symptom in self.symptom_to_idx:
                encoding[self.symptom_to_idx[clean_symptom]] = 1
        return encoding
    
    def create_urgency_keywords(self) -> Dict[str, List[str]]:
        return {
            'critical': ['breathing', 'unconscious', 'bleeding', 'seizure', 'paralysis', 
                        'poisoning', 'trauma', 'bloated', 'pale', 'heatstroke'],
            'urgent': ['blood', 'vomiting blood', 'limping severely', 'swollen', 
                      'difficulty', 'excessive', 'sudden'],
            'moderate': ['vomiting', 'diarrhea', 'lethargy', 'appetite', 'scratching',
                        'discharge', 'coughing', 'sneezing'],
            'low': ['routine', 'checkup', 'vaccination', 'nail', 'grooming', 
                   'consultation', 'advice', 'cleaning']
        }
    
    def extract_text_features(self, description: str) -> Dict[str, float]:
        features = {}
        clean_desc = self.clean_text(description)
        words = clean_desc.split()
        
        features['text_length'] = len(words)
        features['exclamation_count'] = description.count('!')
        features['question_count'] = description.count('?')
        
        urgency_keywords = self.create_urgency_keywords()
        for level, keywords in urgency_keywords.items():
            count = sum(1 for keyword in keywords if keyword in clean_desc)
            features[f'{level}_keyword_count'] = count
        
        time_indicators = ['immediately', 'urgent', 'emergency', 'asap', 'now', 'quickly']
        features['urgency_words'] = sum(1 for word in time_indicators if word in clean_desc)
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        processed_df = df.copy()
        
        if 'symptoms' in df.columns:
            if fit:
                self.build_symptom_vocabulary(df)
            
            symptom_encodings = np.array([self.encode_symptoms(s) for s in df['symptoms']])
            symptom_df = pd.DataFrame(
                symptom_encodings, 
                columns=[f'symptom_{i}' for i in range(symptom_encodings.shape[1])]
            )
            processed_df = pd.concat([processed_df, symptom_df], axis=1)
        
        if 'description' in df.columns:
            text_features = df['description'].apply(self.extract_text_features)
            text_features_df = pd.DataFrame(list(text_features))
            processed_df = pd.concat([processed_df, text_features_df], axis=1)
        
        if 'species' in df.columns:
            if fit:
                processed_df['species_encoded'] = self.species_encoder.fit_transform(df['species'])
            else:
                processed_df['species_encoded'] = self.species_encoder.transform(df['species'])
        
        if 'urgency_level' in df.columns:
            urgency_mapping = {'critical': 4, 'urgent': 3, 'moderate': 2, 'low': 1}
            processed_df['urgency_numeric'] = df['urgency_level'].map(urgency_mapping)
        
        if 'age_years' in df.columns:
            processed_df['age_category_encoded'] = pd.cut(
                df['age_years'], 
                bins=[0, 1, 3, 7, 12, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(2).astype(int)
        
        numeric_features = ['age_years', 'weight_kg', 'wait_time_minutes']
        existing_numeric = [f for f in numeric_features if f in df.columns]
        
        if fit and existing_numeric:
            processed_df[existing_numeric] = self.scaler.fit_transform(df[existing_numeric])
        elif existing_numeric:
            processed_df[existing_numeric] = self.scaler.transform(df[existing_numeric])
        
        return processed_df
    
    def save_preprocessor(self, filepath: str):
        preprocessor_data = {
            'symptom_to_idx': self.symptom_to_idx,
            'idx_to_symptom': self.idx_to_symptom,
            'symptom_vocabulary': list(self.symptom_vocabulary)
        }
        
        with open(filepath, 'w') as f:
            json.dump(preprocessor_data, f, indent=2)
        
        with open(filepath.replace('.json', '_encoders.pkl'), 'wb') as f:
            pickle.dump({
                'species_encoder': self.species_encoder,
                'scaler': self.scaler
            }, f)
    
    def load_preprocessor(self, filepath: str):
        with open(filepath, 'r') as f:
            preprocessor_data = json.load(f)
        
        self.symptom_to_idx = preprocessor_data['symptom_to_idx']
        self.idx_to_symptom = {int(k): v for k, v in preprocessor_data['idx_to_symptom'].items()}
        self.symptom_vocabulary = set(preprocessor_data['symptom_vocabulary'])
        
        with open(filepath.replace('.json', '_encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
            self.species_encoder = encoders['species_encoder']
            self.scaler = encoders['scaler']

def main():
    preprocessor = VeterinaryDataPreprocessor()
    
    print("Loading datasets...")
    train_df = pd.read_csv('data/raw/train_cases.csv')
    val_df = pd.read_csv('data/raw/val_cases.csv')
    test_df = pd.read_csv('data/raw/test_cases.csv')
    
    print("Preprocessing training data...")
    train_processed = preprocessor.preprocess_dataframe(train_df, fit=True)
    
    print("Preprocessing validation data...")
    val_processed = preprocessor.preprocess_dataframe(val_df, fit=False)
    
    print("Preprocessing test data...")
    test_processed = preprocessor.preprocess_dataframe(test_df, fit=False)
    
    print("Saving processed datasets...")
    train_processed.to_csv('data/processed/train_processed.csv', index=False)
    val_processed.to_csv('data/processed/val_processed.csv', index=False)
    test_processed.to_csv('data/processed/test_processed.csv', index=False)
    
    print("Saving preprocessor...")
    preprocessor.save_preprocessor('data/processed/preprocessor_config.json')
    
    print("\nPreprocessing complete!")
    print(f"Training samples: {len(train_processed)}")
    print(f"Validation samples: {len(val_processed)}")
    print(f"Test samples: {len(test_processed)}")
    print(f"Total features: {len(train_processed.columns)}")
    print(f"Symptom vocabulary size: {len(preprocessor.symptom_vocabulary)}")

if __name__ == "__main__":
    main()