import json
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class VeterinaryDataGenerator:
    def __init__(self):
        self.species = ['dog', 'cat', 'rabbit', 'bird', 'hamster', 'guinea pig']
        
        self.dog_breeds = ['Labrador', 'German Shepherd', 'Golden Retriever', 'Bulldog', 'Poodle', 
                           'Beagle', 'Chihuahua', 'Husky', 'Pug', 'Dachshund']
        
        self.cat_breeds = ['Persian', 'Maine Coon', 'Siamese', 'British Shorthair', 'Ragdoll',
                          'Bengal', 'Abyssinian', 'Scottish Fold', 'Sphynx', 'Russian Blue']
        
        self.symptoms = {
            'critical': [
                'difficulty breathing', 'unconscious', 'severe bleeding', 'seizures',
                'pale gums', 'bloated abdomen', 'unable to urinate', 'paralysis',
                'severe trauma', 'eye injury with vision loss', 'poisoning suspected',
                'heatstroke symptoms', 'severe allergic reaction'
            ],
            'urgent': [
                'vomiting blood', 'bloody diarrhea', 'excessive drooling', 'limping severely',
                'eye discharge with squinting', 'difficulty defecating', 'excessive panting',
                'swollen face', 'sudden behavior change', 'moderate bleeding',
                'difficulty swallowing', 'persistent coughing'
            ],
            'moderate': [
                'vomiting', 'diarrhea', 'loss of appetite', 'lethargy', 'minor limping',
                'excessive scratching', 'ear discharge', 'minor eye discharge',
                'sneezing', 'mild coughing', 'drinking more water than usual',
                'urinating more frequently'
            ],
            'low': [
                'bad breath', 'mild scratching', 'nail trimming needed', 'routine checkup',
                'vaccination due', 'weight management consultation', 'dental cleaning inquiry',
                'behavioral consultation', 'diet advice needed', 'grooming required'
            ]
        }
        
        self.age_modifiers = {
            'puppy': (0, 1, 1.2),
            'kitten': (0, 1, 1.2),
            'young': (1, 3, 1.0),
            'adult': (3, 8, 0.9),
            'senior': (8, 15, 1.3),
            'geriatric': (15, 20, 1.5)
        }

    def generate_case(self, case_id: int) -> Dict:
        species = random.choice(self.species)
        
        if species == 'dog':
            breed = random.choice(self.dog_breeds)
            age_category = random.choice(['puppy', 'young', 'adult', 'senior', 'geriatric'])
        elif species == 'cat':
            breed = random.choice(self.cat_breeds)
            age_category = random.choice(['kitten', 'young', 'adult', 'senior', 'geriatric'])
        else:
            breed = 'mixed'
            age_category = random.choice(['young', 'adult', 'senior'])
        
        age_range = self.age_modifiers[age_category]
        age = random.uniform(age_range[0], age_range[1])
        urgency_modifier = age_range[2]
        
        urgency_level = random.choices(
            ['critical', 'urgent', 'moderate', 'low'],
            weights=[5, 15, 40, 40]
        )[0]
        
        symptom_count = random.randint(1, 4)
        primary_symptoms = random.sample(self.symptoms[urgency_level], 
                                       min(symptom_count, len(self.symptoms[urgency_level])))
        
        if random.random() < 0.3 and urgency_level != 'critical':
            other_level = random.choice([k for k in self.symptoms.keys() if k != urgency_level])
            additional = random.sample(self.symptoms[other_level], 1)
            primary_symptoms.extend(additional)
        
        urgency_score = {
            'critical': 5,
            'urgent': 4,
            'moderate': 3,
            'low': 2
        }[urgency_level]
        
        urgency_score = min(5, int(urgency_score * urgency_modifier))
        
        description = self.generate_description(species, breed, age, primary_symptoms)
        
        case = {
            'case_id': case_id,
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
            'species': species,
            'breed': breed,
            'age_years': round(age, 1),
            'age_category': age_category,
            'weight_kg': self.generate_weight(species, breed, age),
            'symptoms': primary_symptoms,
            'description': description,
            'urgency_level': urgency_level,
            'urgency_score': urgency_score,
            'wait_time_minutes': self.calculate_wait_time(urgency_score),
            'requires_immediate_attention': urgency_score >= 4,
            'follow_up_required': random.choice([True, False]),
            'previous_conditions': self.generate_history() if random.random() < 0.4 else []
        }
        
        return case
    
    def generate_description(self, species: str, breed: str, age: float, symptoms: List[str]) -> str:
        templates = [
            f"My {age:.1f} year old {breed} {species} has been experiencing {', '.join(symptoms)}. ",
            f"Patient is a {breed} {species}, {age:.1f} years old, presenting with {', '.join(symptoms)}. ",
            f"Concerned about my {species} ({breed}, {age:.1f}yo) showing signs of {', '.join(symptoms)}. ",
            f"{breed} {species} with {', '.join(symptoms)}. Age: {age:.1f} years. ",
        ]
        
        base = random.choice(templates)
        
        duration = random.choice([
            "Started today.",
            "Has been going on for 2 days.",
            "Noticed yesterday.",
            "Symptoms for the past week.",
            "Just started an hour ago.",
            "Progressive over 3 days."
        ])
        
        concern = random.choice([
            "Very worried.",
            "Please advise urgency.",
            "Should we come in immediately?",
            "Is this an emergency?",
            "Seems to be getting worse.",
            "Pet is distressed."
        ])
        
        return base + duration + " " + concern
    
    def generate_weight(self, species: str, breed: str, age: float) -> float:
        weight_ranges = {
            'dog': {
                'Chihuahua': (2, 6),
                'Pug': (14, 18),
                'Beagle': (20, 30),
                'Labrador': (25, 36),
                'German Shepherd': (30, 40),
                'default': (15, 35)
            },
            'cat': {
                'default': (3.5, 6.5)
            },
            'rabbit': {'default': (1.5, 3.5)},
            'bird': {'default': (0.1, 0.5)},
            'hamster': {'default': (0.1, 0.2)},
            'guinea pig': {'default': (0.7, 1.2)}
        }
        
        if species in weight_ranges:
            if species == 'dog' and breed in weight_ranges['dog']:
                weight_range = weight_ranges['dog'][breed]
            else:
                weight_range = weight_ranges[species]['default']
            
            base_weight = random.uniform(weight_range[0], weight_range[1])
            
            if age < 1:
                base_weight *= 0.5
            elif age < 2:
                base_weight *= 0.8
                
            return round(base_weight, 1)
        
        return round(random.uniform(1, 10), 1)
    
    def calculate_wait_time(self, urgency_score: int) -> int:
        wait_times = {
            5: random.randint(0, 5),
            4: random.randint(5, 15),
            3: random.randint(15, 45),
            2: random.randint(45, 120),
            1: random.randint(120, 240)
        }
        return wait_times.get(urgency_score, 60)
    
    def generate_history(self) -> List[str]:
        conditions = [
            'diabetes', 'arthritis', 'allergies', 'heart murmur', 'kidney disease',
            'dental disease', 'obesity', 'anxiety', 'skin condition', 'eye problems',
            'ear infections', 'hip dysplasia', 'cancer remission', 'thyroid issues'
        ]
        return random.sample(conditions, random.randint(1, 3))
    
    def generate_dataset(self, num_cases: int = 1000) -> pd.DataFrame:
        cases = []
        for i in range(num_cases):
            cases.append(self.generate_case(i + 1))
        
        return pd.DataFrame(cases)

def main():
    generator = VeterinaryDataGenerator()
    
    print("Generating training dataset...")
    train_df = generator.generate_dataset(5000)
    train_df.to_csv('data/raw/train_cases.csv', index=False)
    
    print("Generating validation dataset...")
    val_df = generator.generate_dataset(1000)
    val_df.to_csv('data/raw/val_cases.csv', index=False)
    
    print("Generating test dataset...")
    test_df = generator.generate_dataset(1000)
    test_df.to_csv('data/raw/test_cases.csv', index=False)
    
    with open('data/raw/train_cases.json', 'w') as f:
        json.dump(train_df.to_dict('records'), f, indent=2)
    
    print("\nDataset Statistics:")
    print(f"Total cases generated: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"Training cases: {len(train_df)}")
    print(f"Validation cases: {len(val_df)}")
    print(f"Test cases: {len(test_df)}")
    print(f"\nUrgency Distribution (Training):")
    print(train_df['urgency_level'].value_counts())
    print(f"\nSpecies Distribution (Training):")
    print(train_df['species'].value_counts())

if __name__ == "__main__":
    main()