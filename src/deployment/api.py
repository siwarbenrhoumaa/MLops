from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import mlflow
import dagshub
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration DagsHub + MLflow
dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)

app = FastAPI(
    title="Crime Prediction API",
    description="API de prédiction des crimes de Los Angeles avec MLOps",
    version="2.0.0",
)

# CORS
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
predictions_history = []

# Mapping des classes
CRIME_CLASSES = {
    0: "Other / Fraud / Public Order Crime",
    1: "Property & Theft Crime",
    2: "Vehicle-Related Crime",
    3: "Violent Crime"
}


# ============================================================================
#                           MODÈLES PYDANTIC
# ============================================================================

class CrimeFeatures(BaseModel):
    Hour: int = Field(..., ge=0, le=23, description="Heure (0-23)")
    Day_of_week: int = Field(..., ge=0, le=6, description="Jour (0=Lundi, 6=Dimanche)")
    Month_num: int = Field(..., ge=1, le=12, description="Mois (1-12)")
    LAT: float = Field(..., ge=33.0, le=35.0, description="Latitude")
    LON: float = Field(..., ge=-119.0, le=-117.0, description="Longitude")
    Vict_Age: Optional[float] = Field(None, ge=0, le=120, description="Âge victime")
    AREA: Optional[int] = Field(None, ge=1, le=21, description="Zone")

    class Config:
        schema_extra = {
            "example": {
                "Hour": 20,
                "Day_of_week": 5,
                "Month_num": 7,
                "LAT": 34.0522,
                "LON": -118.2437,
                "Vict_Age": 35,
                "AREA": 12
            }
        }


class PredictionResponse(BaseModel):
    predicted_crime: str
    predicted_crime_code: int
    confidence: float
    timestamp: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    n_predictions: int
    predictions: List[PredictionResponse]
    processing_time: float


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
    metrics: dict
    features: List[str]
    n_classes: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    total_predictions: int


# ============================================================================
#                           FONCTIONS UTILITAIRES
# ============================================================================

def load_model_from_production() -> bool:
    """Charge le modèle depuis MLflow Production"""
    global model, model_version
    
    try:
        model_uri = "models:/crime-prediction-model/Production"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Récupérer la version
        client = mlflow.tracking.MlflowClient()
        prod_versions = client.get_latest_versions("crime-prediction-model", stages=["Production"])
        
        if not prod_versions:
            logger.error("Aucun modèle trouvé en stage Production")
            return False
            
        latest_version = prod_versions[0]
        model = loaded_model
        model_version = latest_version.version
        
        logger.info(f"Modèle chargé avec succès : version {model_version} (run_id: {latest_version.run_id})")
        return True
        
    except Exception as e:
        logger.error(f"Échec du chargement du modèle : {e}")
        model = None
        model_version = None
        return False


def prepare_features(features: CrimeFeatures) -> pd.DataFrame:
    """Prépare les features avec les bons noms de colonnes"""
    feature_dict = {
        'Hour': features.Hour,
        'Day_of_week': features.Day_of_week,
        'Month_num': features.Month_num,
        'LAT': features.LAT,
        'LON': features.LON
    }
    
    if features.Vict_Age is not None:
        feature_dict['Vict Age'] = features.Vict_Age  # Attention à l'espace dans le nom !
    
    if features.AREA is not None:
        feature_dict['AREA'] = features.AREA
    
    return pd.DataFrame([feature_dict])


def log_prediction(features: CrimeFeatures, prediction: int, confidence: float):
    """Log la prédiction pour monitoring"""
    global prediction_count, predictions_history
    
    prediction_count += 1
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': CRIME_CLASSES[prediction],
        'confidence': confidence,
        'features': features.dict()
    }
    
    predictions_history.append(log_entry)
    
    # Garder seulement les 1000 dernières
    if len(predictions_history) > 1000:
        predictions_history = predictions_history[-1000:]


def predict_single(features: CrimeFeatures) -> PredictionResponse:
    """Fonction interne pour une seule prédiction (utilisée par /predict et /batch)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        X = prepare_features(features)
        
        # Prédiction
        pred_code = int(model.predict(X)[0])
        predicted_crime = CRIME_CLASSES[pred_code]
        
        # Confiance
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            except:
                pass
        
        # Log
        log_prediction(features, pred_code, confidence)
        
        return PredictionResponse(
            predicted_crime=predicted_crime,
            predicted_crime_code=pred_code,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version=str(model_version or "unknown")
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction")


# ============================================================================
#                           ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    logger.info("Démarrage de l'API - Chargement du modèle...")
    success = load_model_from_production()
    if not success:
        logger.warning("L'API démarre sans modèle chargé. Utilisez /reload-model pour charger manuellement.")


@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Crime Prediction API v2.0",
        "status": "running",
        "model_loaded": model is not None,
        "model_version": str(model_version) if model_version else "none",
        "endpoints": [
            "/predict", "/predict/batch", "/health", "/model-info",
            "/metrics", "/reload-model", "/docs"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_version=str(model_version) if model_version else None,
        total_predictions=prediction_count
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Info"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions("crime-prediction-model", stages=["Production"])
        if not versions:
            raise HTTPException(status_code=404, detail="Aucun modèle en Production")
        
        version_info = versions[0]
        run = client.get_run(version_info.run_id)
        
        return ModelInfo(
            model_name="crime-prediction-model",
            model_version=str(version_info.version),
            model_stage="Production",
            metrics={
                "test_accuracy": run.data.metrics.get("test_accuracy", 0.0),
                "test_f1": run.data.metrics.get("test_f1", 0.0),
                "cv_mean_accuracy": run.data.metrics.get("cv_mean_accuracy", 0.0)
            },
            features=['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA'],
            n_classes=4
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur MLflow : {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(features: CrimeFeatures):
    return predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prédiction"])
async def predict_batch(features_list: List[CrimeFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    if len(features_list) == 0:
        raise HTTPException(status_code=400, detail="Liste vide")
    
    if len(features_list) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 prédictions par batch")
    
    start_time = time.time()
    predictions = []
    
    # Prédiction vectorisée si possible, sinon boucle optimisée
    try:
        # Préparer toutes les features en un seul DataFrame
        df_list = [prepare_features(f).iloc[0] for f in features_list]
        X_batch = pd.DataFrame(df_list)
        
        # Prédiction batch
        pred_codes = model.predict(X_batch)
        pred_codes = pred_codes.astype(int)
        
        # Confiance batch
        confidences = np.ones(len(pred_codes))
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_batch)
                confidences = np.max(probas, axis=1)
            except:
                pass
        
        # Construire les réponses
        for i, (features, pred_code, conf) in enumerate(zip(features_list, pred_codes, confidences)):
            log_prediction(features, pred_code, float(conf))
            predictions.append(PredictionResponse(
                predicted_crime=CRIME_CLASSES[pred_code],
                predicted_crime_code=pred_code,
                confidence=float(conf),
                timestamp=datetime.now().isoformat(),
                model_version=str(model_version or "unknown")
            ))
            
    except Exception as e:
        logger.error(f"Erreur en batch : {e}")
        # Fallback : prédiction une par une
        for features in features_list:
            predictions.append(predict_single(features))
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        n_predictions=len(predictions),
        predictions=predictions,
        processing_time=round(processing_time, 3)
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    if prediction_count == 0:
        return {"total_predictions": 0, "message": "Aucune prédiction encore"}
    
    df = pd.DataFrame(predictions_history)
    crime_dist = df['prediction'].value_counts(normalize=True).round(3).to_dict()
    
    return {
        "total_predictions": prediction_count,
        "recent_predictions": len(predictions_history),
        "crime_distribution_percentage": crime_dist,
        "average_confidence": round(float(df['confidence'].mean()), 4),
        "last_prediction": predictions_history[-1]['timestamp']
    }


@app.post("/reload-model", tags=["Admin"])
async def reload_model():
    """Recharge le modèle en arrière-plan"""
    def _reload():
        time.sleep(1)  # Petit délai pour éviter les conflits
        load_model_from_production()
    
    # Exécution en background
    import threading
    thread = threading.Thread(target=_reload)
    thread.start()
    
    return {"message": "Rechargement du modèle lancé en arrière-plan", "status": "started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=8000, reload=True)