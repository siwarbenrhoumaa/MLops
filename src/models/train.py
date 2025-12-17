"""
Script d'entraînement des modèles avec tracking MLflow et SMOTE
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # ← Ajout pour gérer les NaN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import argparse
import os
import sys
import warnings
from collections import Counter
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Ajouter le chemin parent pour importer éventuellement du preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------
# Fonction pour regrouper les crimes en 4 classes
# ---------------------------------------------------------
def map_crime_group_4(desc):
    if pd.isna(desc):
        return "Other / Fraud / Public Order Crime"
    desc = str(desc).upper()
    # Violent Crime (incl. Sexual)
    if any(k in desc for k in [
        "ASSAULT", "BATTERY", "ROBBERY", "HOMICIDE",
        "MANSLAUGHTER", "KIDNAPPING", "CRIMINAL THREATS",
        "INTIMATE PARTNER", "RAPE", "SEX", "SODOMY",
        "ORAL COPULATION", "LEWD", "PORNOGRAPHY",
        "FALSE IMPRISONMENT"
    ]):
        return "Violent Crime"

    # Property & Theft Crime
    if any(k in desc for k in [
        "THEFT", "BURGLARY", "SHOPLIFTING",
        "VANDALISM", "ARSON", "PICKPOCKET",
        "PURSE SNATCH", "TRESPASS", "BIKE",
        "ILLEGAL DUMPING"
    ]):
        return "Property & Theft Crime"

    # Vehicle-Related Crime
    if any(k in desc for k in [
        "VEHICLE", "DWOC", "MOTOR VEHICLE",
        "BOAT"
    ]):
        return "Vehicle-Related Crime"

    # Other / Fraud / Public Order Crime
    return "Other / Fraud / Public Order Crime"


# ---------------------------------------------------------
# Fonction pour récupérer le modèle
# ---------------------------------------------------------
def get_model(model_name, params=None):
    if params is None:
        params = {}
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 15),
            min_samples_split=params.get('min_samples_split', 10),
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        ),
        'xgboost': XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'lightgbm': LGBMClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        'logistic_regression': LogisticRegression(
            max_iter=params.get('max_iter', 1000),
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
    }
    return models.get(model_name.lower())


# ---------------------------------------------------------
# Fonction d'entraînement
# ---------------------------------------------------------
def train_model(X_train, X_test, y_train, y_test, model_name, params=None):
    with mlflow.start_run(run_name=f"{model_name}_baseline"):
        mlflow.log_param("model_type", model_name)
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        print(f"\n🚀 Entraînement du modèle : {model_name}")
        model = get_model(model_name, params)

        print("   - Entraînement en cours...")
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)

        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{model_name}_baseline.joblib')
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats pour {model_name} :")
        print(f"   - Train Accuracy : {train_accuracy:.4f}")
        print(f"   - Test Accuracy  : {test_accuracy:.4f}")
        print(f"   - Test F1-Score  : {test_f1:.4f}")
        print(f"   - CV Accuracy    : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("\n   Rapport de classification :")
        print(classification_report(y_test, y_pred_test, zero_division=0))

        return model, {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'cv_mean': cv_scores.mean()
        }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Entraîner un modèle de prédiction de crimes')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'lightgbm', 'logistic_regression'],
                        help='Type de modèle à entraîner')
    parser.add_argument('--data', type=str, default='data/processed/crime_2020_processed.csv',
                        help='Chemin vers les données prétraitées')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion du test set')
    args = parser.parse_args()
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    mlflow.set_experiment("crime-prediction-baseline")

    print(f"\n📂 Chargement des données depuis : {args.data}")
    df = pd.read_csv(args.data)

    # Appliquer le regroupement en 4 classes
    df['Crime_Group'] = df['Crm Cd Desc'].apply(map_crime_group_4)

    feature_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON']
    optional_features = ['Vict Age', 'AREA']
    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)

    X = df[feature_cols].copy()
    y = df['Crime_Group']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"   - Features : {feature_cols}")
    print(f"   - Shape X : {X.shape}")
    print(f"   - Classes : {list(le.classes_)}")
    print(f"   - NaN dans X avant traitement : {X.isna().sum().sum()}")

    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded
    )

    print(f"\n📊 Split des données :")
    print(f"   - Train : {X_train.shape}")
    print(f"   - Test  : {X_test.shape}")

    # 🔧 Imputation des valeurs manquantes (principalement Vict Age)
    print("\n🔧 Imputation des valeurs manquantes...")
    imputer = SimpleImputer(strategy='median')  # Median robuste pour les âges

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"   - NaN restants dans X_train : {X_train.isna().sum().sum()}")
    print(f"   - NaN restants dans X_test  : {X_test.isna().sum().sum()}")

    # ⚖️ Équilibrage avec SMOTE
    print("\n⚖️ Équilibrage des classes avec SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("   - Classes après SMOTE :", Counter(y_train_res))

    # Entraînement
    model, metrics = train_model(X_train_res, X_test, y_train_res, y_test, args.model)

    print(f"\n✅ Modèle {args.model} entraîné et sauvegardé avec succès !")
    print(f"   - Fichier modèle : models/{args.model}_baseline.joblib")
    print(f"   - Suivi MLflow   : Lancez 'mlflow ui' pour visualiser les runs")


if __name__ == "__main__":
    main()