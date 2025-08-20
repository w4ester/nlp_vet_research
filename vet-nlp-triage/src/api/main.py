from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import json
import joblib
import pandas as pd
import numpy as np
import asyncio
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.symptom_extractor import VeterinarySymptomExtractor
from models.urgency_classifier import VeterinaryUrgencyClassifier
from data_prep.preprocessor import VeterinaryDataPreprocessor

app = FastAPI(title="Veterinary Triage API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TriageRequest(BaseModel):
    case_id: Optional[str] = None
    species: str
    breed: Optional[str] = "mixed"
    age_years: float
    weight_kg: Optional[float] = None
    symptoms: List[str]
    description: str
    previous_conditions: Optional[List[str]] = []

class TriageResponse(BaseModel):
    case_id: str
    urgency_score: int
    urgency_level: str
    wait_time_minutes: int
    requires_immediate_attention: bool
    extracted_symptoms: Dict
    confidence_score: float
    recommendations: List[str]
    timestamp: str

class QueueItem(BaseModel):
    case_id: str
    species: str
    urgency_score: int
    urgency_level: str
    wait_time_minutes: int
    timestamp: str
    description: str

class TriageService:
    def __init__(self):
        self.symptom_extractor = VeterinarySymptomExtractor()
        self.urgency_classifier = VeterinaryUrgencyClassifier(model_type='ensemble')
        self.preprocessor = VeterinaryDataPreprocessor()
        self.triage_queue = deque(maxlen=100)
        self.connected_clients = set()
        
        self._load_models()
    
    def _load_models(self):
        try:
            self.urgency_classifier.load_model('data/models/urgency_classifier')
            self.preprocessor.load_preprocessor('data/processed/preprocessor_config.json')
            print("Models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            print("Using untrained models for demonstration")
    
    def process_triage(self, request: TriageRequest) -> TriageResponse:
        case_data = {
            'case_id': request.case_id or f"CASE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'species': request.species,
            'breed': request.breed,
            'age_years': request.age_years,
            'weight_kg': request.weight_kg or self._estimate_weight(request.species, request.age_years),
            'symptoms': request.symptoms,
            'description': request.description,
            'previous_conditions': request.previous_conditions
        }
        
        extracted = self.symptom_extractor.extract_from_case(case_data)
        
        df = pd.DataFrame([case_data])
        try:
            processed_df = self.preprocessor.preprocess_dataframe(df, fit=False)
            urgency_score = self.urgency_classifier.predict(processed_df)[0]
            
            proba = self.urgency_classifier.predict_proba(processed_df)
            if proba is not None:
                confidence = float(np.max(proba[0]))
            else:
                confidence = 0.75
        except:
            urgency_score = min(5, max(1, int(extracted['urgency_score'])))
            confidence = 0.65
        
        urgency_level = self._score_to_level(urgency_score)
        wait_time = self._calculate_wait_time(urgency_score)
        
        recommendations = self._generate_recommendations(urgency_score, extracted)
        
        response = TriageResponse(
            case_id=case_data['case_id'],
            urgency_score=urgency_score,
            urgency_level=urgency_level,
            wait_time_minutes=wait_time,
            requires_immediate_attention=(urgency_score >= 4),
            extracted_symptoms=extracted,
            confidence_score=confidence,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        queue_item = QueueItem(
            case_id=response.case_id,
            species=request.species,
            urgency_score=urgency_score,
            urgency_level=urgency_level,
            wait_time_minutes=wait_time,
            timestamp=response.timestamp,
            description=request.description[:100]
        )
        self.triage_queue.appendleft(queue_item)
        
        return response
    
    def _estimate_weight(self, species: str, age: float) -> float:
        weights = {
            'dog': 25.0,
            'cat': 4.5,
            'rabbit': 2.0,
            'bird': 0.3,
            'hamster': 0.15,
            'guinea pig': 1.0
        }
        base_weight = weights.get(species.lower(), 5.0)
        
        if age < 1:
            base_weight *= 0.5
        elif age < 2:
            base_weight *= 0.8
        
        return base_weight
    
    def _score_to_level(self, score: int) -> str:
        levels = {
            5: 'critical',
            4: 'urgent',
            3: 'moderate',
            2: 'low',
            1: 'routine'
        }
        return levels.get(score, 'moderate')
    
    def _calculate_wait_time(self, score: int) -> int:
        wait_times = {
            5: 0,
            4: 10,
            3: 30,
            2: 60,
            1: 120
        }
        return wait_times.get(score, 45)
    
    def _generate_recommendations(self, urgency_score: int, extracted: Dict) -> List[str]:
        recommendations = []
        
        if urgency_score >= 4:
            recommendations.append("Immediate veterinary attention required")
            recommendations.append("Call ahead to notify clinic of arrival")
            recommendations.append("Do not give food or water unless instructed")
        elif urgency_score == 3:
            recommendations.append("Schedule appointment within 24 hours")
            recommendations.append("Monitor symptoms closely")
            recommendations.append("Document any changes in condition")
        else:
            recommendations.append("Schedule routine appointment")
            recommendations.append("Continue normal care routine")
            recommendations.append("Keep monitoring for any changes")
        
        if 'respiratory' in extracted.get('symptom_categories', []):
            recommendations.append("Keep pet calm and minimize activity")
        
        if 'gastrointestinal' in extracted.get('symptom_categories', []):
            recommendations.append("Withhold food for 12 hours if vomiting")
        
        return recommendations
    
    def get_queue(self) -> List[QueueItem]:
        return sorted(
            list(self.triage_queue),
            key=lambda x: x.urgency_score,
            reverse=True
        )

triage_service = TriageService()

@app.get("/")
async def root():
    return {
        "message": "Veterinary Triage API",
        "version": "1.0.0",
        "endpoints": [
            "/triage",
            "/queue",
            "/health",
            "/ws"
        ]
    }

@app.post("/triage", response_model=TriageResponse)
async def triage_case(request: TriageRequest):
    try:
        response = triage_service.process_triage(request)
        
        for client in triage_service.connected_clients:
            await client.send_json({
                "type": "new_case",
                "data": response.dict()
            })
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue")
async def get_triage_queue():
    queue = triage_service.get_queue()
    return {
        "total_cases": len(queue),
        "queue": [item.dict() for item in queue]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": True,
        "queue_size": len(triage_service.triage_queue)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    triage_service.connected_clients.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to triage system"
        })
        
        while True:
            data = await websocket.receive_text()
            
            if data == "get_queue":
                queue = triage_service.get_queue()
                await websocket.send_json({
                    "type": "queue_update",
                    "data": [item.dict() for item in queue]
                })
    
    except WebSocketDisconnect:
        triage_service.connected_clients.remove(websocket)

@app.post("/analyze")
async def analyze_symptoms(text: str):
    extracted = triage_service.symptom_extractor.extract_symptoms(text)
    return {
        "extracted_symptoms": extracted,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)