"""
Backend FastAPI pour la prédiction de crimes à Los Angeles
Version Universelle - Compatible avec TOUS les types de modèles
Auto-installation des dépendances manquantes
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import mlflow
import dagshub
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import OrdinalEncoder
import asyncio
import os
import subprocess
import sys

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INSTALLATION AUTOMATIQUE DES DÉPENDANCES
# ============================================================================
def install_missing_dependencies(model_uri):
    """
    Installe automatiquement les dépendances manquantes du modèle
    """
    try:
        logger.info("🔍 Vérification des dépendances du modèle...")
        
        # Récupérer les dépendances requises
        requirements = mlflow.pyfunc.get_model_dependencies(model_uri)
        
        if requirements:
            logger.info(f"📦 Installation des dépendances manquantes...")
            
            # Créer un fichier temporaire avec les requirements
            with open('/tmp/model_requirements.txt', 'w') as f:
                f.write(requirements)
            
            # Installer les dépendances
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                '-r', '/tmp/model_requirements.txt',
                '--quiet'
            ])
            
            logger.info("✅ Dépendances installées avec succès")
            return True
        else:
            logger.info("✅ Toutes les dépendances sont déjà installées")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ Impossible d'installer automatiquement les dépendances : {e}")
        return False


# ============================================================================
# CHARGEMENT UNIVERSEL DE MODÈLE
# ============================================================================
def load_model_universal(model_uri):
    """
    Charge un modèle de manière universelle (sklearn, joblib, pickle, etc.)
    Compatible avec tous les types de modèles
    """
    try:
        # Méthode 1 : MLflow PyFunc (universel)
        logger.info("📥 Tentative de chargement via MLflow PyFunc...")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Modèle chargé via PyFunc")
        return model, "pyfunc"
        
    except Exception as e1:
        logger.warning(f"⚠️ Échec PyFunc : {e1}")
        
        try:
            # Méthode 2 : MLflow Sklearn
            logger.info("📥 Tentative de chargement via MLflow Sklearn...")
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Modèle chargé via Sklearn")
            return model, "sklearn"
            
        except Exception as e2:
            logger.warning(f"⚠️ Échec Sklearn : {e2}")
            
            try:
                # Méthode 3 : Télécharger et charger le .joblib directement
                logger.info("📥 Tentative de chargement du .joblib direct...")
                import tempfile
                import joblib
                
                client = mlflow.tracking.MlflowClient()
                
                # Extraire le run_id depuis l'URI
                if "models:/" in model_uri:
                    # Format: models:/model_name/stage
                    parts = model_uri.split("/")
                    model_name = parts[1]
                    stage = parts[2] if len(parts) > 2 else "Production"
                    
                    versions = client.get_latest_versions(model_name, stages=[stage])
                    if versions:
                        run_id = versions[0].run_id
                    else:
                        raise ValueError(f"Aucune version trouvée pour {model_name}/{stage}")
                else:
                    # Format: runs:/run_id/artifact_path
                    run_id = model_uri.split("/")[1]
                
                # Lister les artifacts
                artifacts = client.list_artifacts(run_id)
                
                # Chercher un fichier .joblib
                joblib_files = [art.path for art in artifacts if art.path.endswith('.joblib')]
                
                if joblib_files:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Télécharger le premier .joblib trouvé
                        local_path = client.download_artifacts(run_id, joblib_files[0], dst_path=tmpdir)
                        full_path = os.path.join(tmpdir, joblib_files[0])
                        
                        # Charger
                        model = joblib.load(full_path)
                        logger.info(f"✅ Modèle chargé depuis {joblib_files[0]}")
                        return model, "joblib"
                else:
                    raise FileNotFoundError("Aucun fichier .joblib trouvé")
                    
            except Exception as e3:
                logger.error(f"❌ Échec de toutes les méthodes de chargement")
                logger.error(f"   PyFunc: {e1}")
                logger.error(f"   Sklearn: {e2}")
                logger.error(f"   Joblib: {e3}")
                raise


# Initialisation DagsHub + MLflow
try:
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    logger.info("✅ Connexion DagsHub/MLflow réussie")
except Exception as e:
    logger.error(f"❌ Erreur connexion DagsHub : {e}")

# Initialisation FastAPI
app = FastAPI(
    title="Crime Prediction API - Los Angeles",
    description="Prédiction du groupe de crime (4 classes) - Modèle Universel avec Auto-Install",
    version="3.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================
model = None
model_version = None
model_run_id = None
model_type = None  # Type du modèle (pyfunc, sklearn, joblib)
model_metrics = {}
prediction_count = 0
ordinal_encoder = None
last_check_time = None

CRIME_CLASSES = {
    0: "Other / Fraud / Public Order Crime",
    1: "Property & Theft Crime",
    2: "Vehicle-Related Crime",
    3: "Violent Crime"
}

# ============================================================================
# MODÈLES PYDANTIC (identiques)
# ============================================================================
class CrimeInput(BaseModel):
    Hour: int = Field(..., ge=0, le=23, description="Heure du crime (0-23)")
    Day_of_week: int = Field(..., ge=0, le=6, description="Jour de la semaine")
    Month_num: int = Field(..., ge=1, le=12, description="Mois (1-12)")
    LAT: float = Field(..., ge=33.0, le=35.0, description="Latitude")
    LON: float = Field(..., ge=-119.0, le=-117.0, description="Longitude")
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
    model_type: Optional[str] = None
    model_run_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_predictions: int
    predictions: List[PredictionResponse]
    processing_time_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    run_id: Optional[str]
    model_type: Optional[str]
    stage: str
    metrics: Dict
    features_used: List[str]
    classes: List[str]
    last_updated: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    model_type: Optional[str]
    model_run_id: Optional[str]
    total_predictions: int
    last_check: Optional[str]


class ReloadResponse(BaseModel):
    status: str
    message: str
    old_version: Optional[str]
    new_version: Optional[str]
    model_type: Optional[str]
    model_run_id: Optional[str]
    timestamp: str


# ============================================================================
# CHARGEMENT DU MODÈLE (VERSION UNIVERSELLE)
# ============================================================================
def load_production_model():
    """
    Charge le modèle en production depuis MLflow
    Version universelle compatible avec tous les types de modèles
    """
    global model, model_version, model_run_id, model_type, ordinal_encoder, model_metrics, last_check_time
    
    logger.info("🔄 Chargement du modèle Production...")
    
    try:
        model_uri = "models:/crime-prediction-model/Production"
        
        # ÉTAPE 1 : Installer les dépendances si nécessaire
        install_missing_dependencies(model_uri)
        
        # ÉTAPE 2 : Charger le modèle de manière universelle
        loaded_model, load_type = load_model_universal(model_uri)
        
        # ÉTAPE 3 : Récupérer les métadonnées
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("crime-prediction-model", stages=["Production"])
        
        if not versions:
            logger.error("❌ Aucun modèle en Production trouvé")
            return False
        
        v = versions[0]
        old_version = model_version
        
        # ÉTAPE 4 : Mettre à jour les variables globales
        model = loaded_model
        model_version = v.version
        model_run_id = v.run_id
        model_type = load_type
        last_check_time = datetime.now()
        
        # ÉTAPE 5 : Récupérer les métriques
        try:
            run = client.get_run(v.run_id)
            model_metrics = {
                "test_accuracy": run.data.metrics.get("test_accuracy", 0),
                "test_f1_weighted": run.data.metrics.get("test_f1_weighted", 0),
                "cv_accuracy_mean": run.data.metrics.get("cv_accuracy_mean", 0),
            }
            
            # Déterminer le type de modèle depuis les params
            model_name_from_params = run.data.params.get('model_type') or run.data.params.get('ensemble_type', 'unknown')
            logger.info(f"📊 Type de modèle : {model_name_from_params}")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de récupérer les métriques : {e}")
            model_metrics = {}
        
        # ÉTAPE 6 : Initialiser l'encodeur catégoriel
        ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', 
            unknown_value=-1
        )
        
        sample_data = pd.DataFrame([
            ['M', 'H'], ['F', 'B'], ['X', 'W'],
            ['Unknown', 'A'], ['Unknown', 'O'], ['Unknown', 'Unknown']
        ], columns=['Vict Sex', 'Vict Descent'])
        
        ordinal_encoder.fit(sample_data)
        
        # ÉTAPE 7 : Log du changement
        if old_version and old_version != model_version:
            logger.info(f"🔄 Modèle mis à jour : v{old_version} → v{model_version}")
        else:
            logger.info(f"✅ Modèle chargé : version {model_version}")
        
        logger.info(f"   Type : {load_type}")
        logger.info(f"   Run ID : {model_run_id[:12]}...")
        logger.info(f"   Test Accuracy : {model_metrics.get('test_accuracy', 0):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Échec du chargement du modèle : {e}")
        import traceback
        traceback.print_exc()
        
        model = None
        model_version = None
        model_run_id = None
        model_type = None
        ordinal_encoder = None
        return False


# ============================================================================
# PRÉPARATION DES DONNÉES (identique)
# ============================================================================
def prepare_input(input_data: CrimeInput) -> pd.DataFrame:
    """Prépare les données d'entrée pour la prédiction"""
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
    
    # Imputation numérique
    numeric = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA', 'Premis Cd', 'Part 1-2']
    for col in numeric:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Encodage catégorique
    cat = ['Vict Sex', 'Vict Descent']
    if ordinal_encoder:
        try:
            encoded = ordinal_encoder.transform(df[cat])
            df[cat] = pd.DataFrame(encoded, columns=cat, index=df.index).astype('int64')
        except Exception as e:
            logger.warning(f"⚠️ Erreur encodage : {e}")
            df[cat] = 0
    
    return df


# ============================================================================
# ENDPOINTS (identiques mais avec model_type ajouté)
# ============================================================================
@app.on_event("startup")
async def startup():
    """Événement de démarrage"""
    logger.info("🚀 Démarrage de l'API Crime Prediction...")
    
    success = load_production_model()
    
    if not success:
        logger.warning("⚠️ Aucun modèle chargé au démarrage")
    
    logger.info("✅ API prête")


@app.get("/")
def root():
    """Page d'accueil de l'API"""
    return {
        "service": "Crime Prediction API",
        "version": "3.1.0",
        "model_loaded": model is not None,
        "model_version": model_version,
        "model_type": model_type,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict/batch",
            "model_info": "/model-info",
            "reload": "/reload-model"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Vérification de l'état de santé de l'API"""
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        model_type=model_type,
        model_run_id=model_run_id,
        total_predictions=prediction_count,
        last_check=last_check_time.isoformat() if last_check_time else None
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Informations détaillées sur le modèle en production"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("crime-prediction-model", stages=["Production"])
        
        if not versions:
            raise HTTPException(status_code=503, detail="Aucun modèle en Production")
        
        v = versions[0]
        
        return ModelInfoResponse(
            model_name="crime-prediction-model",
            version=v.version,
            run_id=v.run_id,
            model_type=model_type,
            stage="Production",
            metrics=model_metrics,
            features_used=['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA',
                           'Vict Sex', 'Vict Descent', 'Premis Cd', 'Part 1-2'],
            classes=list(CRIME_CLASSES.values()),
            last_updated=datetime.fromtimestamp(v.last_updated_timestamp/1000).isoformat()
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur récupération infos : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: CrimeInput):
    """Prédiction individuelle"""
    global prediction_count
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        X = prepare_input(input_data)
        pred = int(model.predict(X)[0])
        crime = CRIME_CLASSES.get(pred, "Unknown")
        
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            except:
                pass
        
        prediction_count += 1
        
        return PredictionResponse(
            predicted_crime_group=crime,
            predicted_class_code=pred,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version=model_version or "unknown",
            model_type=model_type,
            model_run_id=model_run_id
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(inputs: List[CrimeInput]):
    """Prédiction par batch"""
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
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                except:
                    pass
            
            prediction_count += 1
            
            results.append(PredictionResponse(
                predicted_crime_group=crime,
                predicted_class_code=pred,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                model_version=model_version or "unknown",
                model_type=model_type,
                model_run_id=model_run_id
            ))
        
        duration = (datetime.now() - start).total_seconds()
        
        return BatchPredictionResponse(
            total_predictions=len(results),
            predictions=results,
            processing_time_seconds=round(duration, 3)
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur batch : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur batch : {str(e)}")


@app.post("/reload-model", response_model=ReloadResponse)
def reload_model(background_tasks: BackgroundTasks):
    """Recharge le modèle depuis MLflow Production"""
    old_version = model_version
    old_type = model_type
    
    logger.info("🔄 Reload manuel demandé...")
    
    success = load_production_model()
    
    if success:
        return ReloadResponse(
            status="success",
            message="Modèle rechargé avec succès",
            old_version=old_version,
            new_version=model_version,
            model_type=model_type,
            model_run_id=model_run_id,
            timestamp=datetime.now().isoformat()
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Échec du rechargement du modèle"
        )


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )