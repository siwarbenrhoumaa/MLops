"""
Backend FastAPI pour la prédiction de crimes à Los Angeles
Version finale – Modèle chargé correctement au démarrage
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow
import dagshub
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import OrdinalEncoder

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation DagsHub + MLflow
dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)

app = FastAPI(
    title="Crime Prediction API - Los Angeles",
    description="Prédiction du groupe de crime (4 classes) - Modèle MLflow Production",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
model_version = None
prediction_count = 0
ordinal_encoder = None

CRIME_CLASSES = {
    0: "Other / Fraud / Public Order Crime",
    1: "Property & Theft Crime",
    2: "Vehicle-Related Crime",
    3: "Violent Crime"
}

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================
class CrimeInput(BaseModel):
    Hour: int = Field(..., ge=0, le=23)
    Day_of_week: int = Field(..., ge=0, le=6)
    Month_num: int = Field(..., ge=1, le=12)
    LAT: float = Field(..., ge=33.0, le=35.0)
    LON: float = Field(..., ge=-119.0, le=-117.0)
    Vict_Age: Optional[float] = Field(None, ge=0, le=120)
    AREA: Optional[int] = Field(None, ge=1, le=21)
    Vict_Sex: Optional[str] = Field(None, pattern="^(M|F|X)?$")
    Vict_Descent: Optional[str] = Field(None, pattern="^(H|B|W|A|O)?$")
    Premis_Cd: Optional[float] = Field(None)
    Part_1_2: Optional[int] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "Hour": 17,
                "Day_of_week": 5,
                "Month_num": 3,
                "LAT": 34.0522,
                "LON": -118.2437,
                "Vict_Age": 26,
                "AREA": 10
            }
        }

class PredictionResponse(BaseModel):
    predicted_crime_group: str
    predicted_class_code: int
    confidence: float
    timestamp: str
    model_version: str

class BatchPredictionResponse(BaseModel):
    total_predictions: int
    predictions: List[PredictionResponse]
    processing_time_seconds: float

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    metrics: dict
    features_used: List[str]
    classes: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    total_predictions: int

# ============================================================================
# CHARGEMENT DU MODÈLE (CORRIGÉ)
# ============================================================================
def load_production_model():
    global model, model_version, ordinal_encoder
    
    logger.info("Chargement du modèle Production...")
    
    try:
        model_uri = "models:/crime-prediction-model/Production"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("crime-prediction-model", stages=["Production"])
        
        if not versions:
            logger.error("Aucun modèle en Production trouvé")
            return False
        
        v = versions[0]
        model = loaded_model
        model_version = v.version
        
        # === CORRECTION DÉFINITIVE DE L'ENCODEUR ===
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Fit sur 2 colonnes simultanément
        sample_data = pd.DataFrame([
            ['M', 'H'],
            ['F', 'B'],
            ['X', 'W'],
            ['Unknown', 'A'],
            ['Unknown', 'O'],
            ['Unknown', 'Unknown']
        ], columns=['Vict Sex', 'Vict Descent'])
        
        ordinal_encoder.fit(sample_data)
        
        logger.info(f"Modèle chargé avec succès : version {model_version}")
        return True
        
    except Exception as e:
        logger.error(f"Échec du chargement du modèle : {e}")
        model = None
        model_version = None
        ordinal_encoder = None
        return False

# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================
def prepare_input(input_data: CrimeInput) -> pd.DataFrame:
    data = {
        'Hour': [input_data.Hour],
        'Day_of_week': [input_data.Day_of_week],
        'Month_num': [input_data.Month_num],
        'LAT': [input_data.LAT],
        'LON': [input_data.LON],
        'Vict Age': [input_data.Vict_Age if input_data.Vict_Age is not None else np.nan],
        'AREA': [input_data.AREA if input_data.AREA is not None else np.nan],
        'Vict Sex': [input_data.Vict_Sex if input_data.Vict_Sex else 'Unknown'],
        'Vict Descent': [input_data.Vict_Descent if input_data.Vict_Descent else 'Unknown'],
        'Premis Cd': [input_data.Premis_Cd if input_data.Premis_Cd is not None else np.nan],
        'Part 1-2': [input_data.Part_1_2 if input_data.Part_1_2 is not None else np.nan]
    }
    
    df = pd.DataFrame(data)
    
    columns_order = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA',
                     'Vict Sex', 'Vict Descent', 'Premis Cd', 'Part 1-2']
    
    df = df.reindex(columns=columns_order)
    
    # Imputation numérique (médiane)
    numeric = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA', 'Premis Cd', 'Part 1-2']
    df[numeric] = df[numeric].fillna(df[numeric].median())
    
    # Encodage catégorique + FORÇAGE EN INT
    cat = ['Vict Sex', 'Vict Descent']
    if ordinal_encoder:
        encoded = ordinal_encoder.transform(df[cat])
        df[cat] = pd.DataFrame(encoded, columns=cat, index=df.index).astype('int64')  # ← FORÇAGE INT
    
    return df

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.on_event("startup")
async def startup():
    load_production_model()

@app.get("/")
def root():
    return {"service": "Crime Prediction API", "model_loaded": model is not None, "version": model_version}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        total_predictions=prediction_count
    )

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("crime-prediction-model", stages=["Production"])
    v = versions[0]
    run = client.get_run(v.run_id)
    
    return ModelInfoResponse(
        model_name="crime-prediction-model",
        version=v.version,
        stage="Production",
        metrics=run.data.metrics,
        features_used=['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA',
                       'Vict Sex', 'Vict Descent', 'Premis Cd', 'Part 1-2'],
        classes=list(CRIME_CLASSES.values())
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: CrimeInput):
    global prediction_count
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        X = prepare_input(input_data)
        pred = int(model.predict(X)[0])
        crime = CRIME_CLASSES.get(pred, "Unknown")
        
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        
        prediction_count += 1
        
        return PredictionResponse(
            predicted_crime_group=crime,
            predicted_class_code=pred,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version=model_version or "unknown"
        )
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(inputs: List[CrimeInput]):
    global prediction_count
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    start = datetime.now()
    results = []
    
    try:
        for inp in inputs:
            X = prepare_input(inp)
            pred = int(model.predict(X)[0])
            crime = CRIME_CLASSES.get(pred, "Unknown")
            
            confidence = 1.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            
            prediction_count += 1
            results.append(PredictionResponse(
                predicted_crime_group=crime,
                predicted_class_code=pred,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                model_version=model_version or "unknown"
            ))
        
        duration = (datetime.now() - start).total_seconds()
        
        return BatchPredictionResponse(
            total_predictions=len(results),
            predictions=results,
            processing_time_seconds=round(duration, 3)
        )
    except Exception as e:
        logger.error(f"Erreur batch : {e}")
        raise HTTPException(status_code=500, detail="Erreur batch")

@app.post("/reload-model")
def reload_model():
    success = load_production_model()
    return {"status": "success" if success else "failed", "model_version": model_version or "none"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=8000, reload=True)