import re
import json
import spacy
from typing import List, Dict, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict

class VeterinarySymptomExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.symptom_patterns = self.build_symptom_patterns()
        self.body_parts = self.build_body_parts()
        self.modifiers = self.build_modifiers()
        
    def build_symptom_patterns(self) -> Dict[str, List[str]]:
        return {
            'respiratory': [
                'breathing', 'breath', 'cough', 'coughing', 'wheeze', 'wheezing',
                'pant', 'panting', 'gasp', 'gasping', 'choke', 'choking',
                'sneeze', 'sneezing', 'nasal discharge', 'respiratory'
            ],
            'gastrointestinal': [
                'vomit', 'vomiting', 'diarrhea', 'constipation', 'bloat', 'bloated',
                'appetite', 'eating', 'drinking', 'thirst', 'nausea', 'drool',
                'drooling', 'salivate', 'stomach', 'abdomen', 'belly'
            ],
            'neurological': [
                'seizure', 'seizing', 'tremor', 'trembling', 'shake', 'shaking',
                'paralysis', 'paralyzed', 'collapse', 'collapsed', 'unconscious',
                'disoriented', 'confused', 'dizzy', 'balance', 'coordination'
            ],
            'musculoskeletal': [
                'limp', 'limping', 'lame', 'lameness', 'stiff', 'stiffness',
                'swollen', 'swelling', 'joint', 'muscle', 'bone', 'fracture',
                'injury', 'injured', 'pain', 'painful', 'tender'
            ],
            'dermatological': [
                'itch', 'itching', 'scratch', 'scratching', 'rash', 'skin',
                'fur', 'hair', 'bald', 'alopecia', 'lesion', 'wound', 'cut',
                'bite', 'abscess', 'lump', 'bump', 'growth'
            ],
            'ocular': [
                'eye', 'eyes', 'vision', 'blind', 'blindness', 'discharge',
                'squint', 'squinting', 'tear', 'tearing', 'red', 'swollen',
                'cloudy', 'pupil', 'cornea'
            ],
            'behavioral': [
                'aggressive', 'aggression', 'anxious', 'anxiety', 'lethargy',
                'lethargic', 'hide', 'hiding', 'restless', 'pacing', 'vocalize',
                'crying', 'whining', 'behavior', 'personality'
            ],
            'urinary': [
                'urine', 'urinate', 'urinating', 'urination', 'pee', 'peeing',
                'bladder', 'kidney', 'blood in urine', 'straining', 'frequent',
                'accident', 'incontinence'
            ]
        }
    
    def build_body_parts(self) -> Set[str]:
        return {
            'head', 'eye', 'eyes', 'ear', 'ears', 'nose', 'mouth', 'teeth',
            'tooth', 'gum', 'gums', 'tongue', 'throat', 'neck', 'chest',
            'back', 'spine', 'tail', 'leg', 'legs', 'paw', 'paws', 'foot',
            'feet', 'nail', 'nails', 'claw', 'claws', 'stomach', 'abdomen',
            'belly', 'side', 'hip', 'hips', 'joint', 'joints'
        }
    
    def build_modifiers(self) -> Dict[str, List[str]]:
        return {
            'severity': ['mild', 'moderate', 'severe', 'extreme', 'slight'],
            'duration': ['sudden', 'gradual', 'chronic', 'acute', 'persistent'],
            'frequency': ['occasional', 'frequent', 'constant', 'intermittent'],
            'quality': ['sharp', 'dull', 'throbbing', 'burning', 'stabbing']
        }
    
    def extract_symptoms(self, text: str) -> Dict[str, any]:
        text_lower = text.lower()
        doc = self.nlp(text_lower)
        
        extracted = {
            'raw_symptoms': [],
            'symptom_categories': [],
            'body_parts_affected': [],
            'severity_indicators': [],
            'temporal_indicators': [],
            'symptom_descriptions': []
        }
        
        for category, patterns in self.symptom_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    extracted['raw_symptoms'].append(pattern)
                    if category not in extracted['symptom_categories']:
                        extracted['symptom_categories'].append(category)
        
        for body_part in self.body_parts:
            if body_part in text_lower:
                extracted['body_parts_affected'].append(body_part)
        
        severity_keywords = ['severe', 'extreme', 'mild', 'moderate', 'serious', 'critical']
        for keyword in severity_keywords:
            if keyword in text_lower:
                extracted['severity_indicators'].append(keyword)
        
        temporal_keywords = ['suddenly', 'gradually', 'today', 'yesterday', 'hours', 'days', 'weeks']
        for keyword in temporal_keywords:
            if keyword in text_lower:
                extracted['temporal_indicators'].append(keyword)
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    symptom_phrase = self.extract_symptom_phrase(sent, token)
                    if symptom_phrase:
                        extracted['symptom_descriptions'].append(symptom_phrase)
        
        extracted['symptom_count'] = len(extracted['raw_symptoms'])
        extracted['has_multiple_symptoms'] = len(extracted['raw_symptoms']) > 1
        extracted['urgency_score'] = self.calculate_urgency_score(extracted)
        
        return extracted
    
    def extract_symptom_phrase(self, sent, verb_token) -> str:
        phrase_tokens = [verb_token]
        
        for child in verb_token.children:
            if child.dep_ in ['nsubj', 'dobj', 'amod', 'advmod', 'prep', 'pobj']:
                phrase_tokens.append(child)
        
        phrase_tokens.sort(key=lambda x: x.i)
        phrase = ' '.join([token.text for token in phrase_tokens])
        
        return phrase if len(phrase_tokens) > 1 else None
    
    def calculate_urgency_score(self, extracted_symptoms: Dict) -> float:
        score = 0.0
        
        critical_symptoms = ['seizure', 'unconscious', 'breathing', 'paralysis', 'bloat']
        urgent_symptoms = ['vomiting blood', 'severe pain', 'unable to urinate', 'eye injury']
        
        for symptom in extracted_symptoms['raw_symptoms']:
            if any(crit in symptom for crit in critical_symptoms):
                score += 2.0
            elif any(urg in symptom for urg in urgent_symptoms):
                score += 1.5
            else:
                score += 0.5
        
        if 'severe' in extracted_symptoms['severity_indicators'] or 'extreme' in extracted_symptoms['severity_indicators']:
            score *= 1.5
        
        if 'suddenly' in extracted_symptoms['temporal_indicators']:
            score *= 1.2
        
        return min(score, 5.0)
    
    def extract_from_case(self, case: Dict) -> Dict:
        description = case.get('description', '')
        symptoms_list = case.get('symptoms', [])
        
        if isinstance(symptoms_list, str):
            symptoms_list = eval(symptoms_list)
        
        text_extracted = self.extract_symptoms(description)
        
        text_extracted['listed_symptoms'] = symptoms_list
        text_extracted['case_id'] = case.get('case_id')
        text_extracted['species'] = case.get('species')
        text_extracted['age'] = case.get('age_years')
        
        return text_extracted

def main():
    extractor = VeterinarySymptomExtractor()
    
    print("Loading test cases...")
    test_df = pd.read_csv('data/raw/test_cases.csv')
    
    print("\nExtracting symptoms from first 10 cases:")
    print("="*60)
    
    for idx in range(min(10, len(test_df))):
        case = test_df.iloc[idx].to_dict()
        extracted = extractor.extract_from_case(case)
        
        print(f"\nCase {idx + 1}:")
        print(f"Species: {extracted['species']}, Age: {extracted['age']}")
        print(f"Description: {case['description'][:100]}...")
        print(f"Listed symptoms: {extracted['listed_symptoms']}")
        print(f"Extracted symptoms: {extracted['raw_symptoms']}")
        print(f"Categories: {extracted['symptom_categories']}")
        print(f"Body parts: {extracted['body_parts_affected']}")
        print(f"Urgency score: {extracted['urgency_score']:.2f}")
        print("-"*40)
    
    print("\nProcessing all test cases...")
    all_extractions = []
    for idx, row in test_df.iterrows():
        extracted = extractor.extract_from_case(row.to_dict())
        all_extractions.append(extracted)
    
    extraction_df = pd.DataFrame(all_extractions)
    extraction_df.to_csv('data/processed/extracted_symptoms.csv', index=False)
    
    print(f"\nExtraction complete! Processed {len(all_extractions)} cases")
    print(f"Average symptoms per case: {extraction_df['symptom_count'].mean():.2f}")
    print(f"Average urgency score: {extraction_df['urgency_score'].mean():.2f}")

if __name__ == "__main__":
    main()