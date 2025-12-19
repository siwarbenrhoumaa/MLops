"""
Script d'entraînement des modèles avec tracking MLflow et SMOTE
Version finale corrigée - Compatible avec crime_2020_processed2.csv
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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import argparse
import os
import sys
import warnings
import argparse

warnings.filterwarnings('ignore')

# Ajouter le chemin parent si besoin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------
# Fonction pour récupérer le modèle (hyperparamètres originaux)
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
            verbosity=0
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
    return models.get(model_name.lower(), RandomForestClassifier(random_state=42))


# ---------------------------------------------------------
# Fonction d'entraînement
# ---------------------------------------------------------
def train_model(X_train, X_test, y_train, y_test, model_name, label_encoder, feature_cols):
    with mlflow.start_run(run_name=f"{model_name}_2020_baseline"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("target", "Crime_Group")
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        mlflow.log_param("features", ", ".join(feature_cols))

        print(f"\n🚀 Entraînement du modèle : {model_name.upper()}")

        model = get_model(model_name)

        print("   - Fitting en cours...")
        model.fit(X_train, y_train)

        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Métriques
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')

        # Logging MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_weighted", test_f1)
        mlflow.log_metric("test_precision_weighted", test_precision)
        mlflow.log_metric("test_recall_weighted", test_recall)

        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

        # Sauvegarde modèle + artifacts
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f'{model_name}_2020_baseline.joblib')
        joblib.dump(model, model_path)

        artifacts = {
            'model': model,
            'label_encoder': label_encoder,
            'features': feature_cols
        }
        artifacts_path = os.path.join(model_dir, f'{model_name}_artifacts.joblib')
        joblib.dump(artifacts, artifacts_path)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(artifacts_path)

        print(f"\n📊 Résultats {model_name.upper()} :")
        print(f"   - Train Accuracy : {train_acc:.4f}")
        print(f"   - Test Accuracy  : {test_acc:.4f}")
        print(f"   - Test F1-Score  : {test_f1:.4f}")
        print(f"   - CV Accuracy    : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print("\n   Rapport de classification :")
        print(classification_report(y_test, y_pred_test,
                                    target_names=label_encoder.classes_,
                                    zero_division=0))

        return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Entraîner un modèle de prédiction de Crime_Group')
    parser.add_argument('--data', type=str, default='data/processed/crime_2020_processed2.csv')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'lightgbm', 'logistic_regression'],
                        help='Type de modèle à entraîner')
    parser.add_argument('--data', type=str, default='data/processed/crime_2020_processed2.csv',
                        help='Chemin vers les données prétraitées')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion du test set')
    args = parser.parse_args()

    # Initialisation DagsHub + MLflow
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    mlflow.set_experiment("crime-prediction-2020")

    print(f"\n📂 Chargement des données depuis : {args.data}")
    if not os.path.exists(args.data):
        print(f"❌ Fichier introuvable ! Vérifie le chemin.")
        return

    df = pd.read_csv(args.data)

    if 'Crime_Group' not in df.columns:
        print("❌ 'Crime_Group' manquant → relance preprocessed.py")
        return

    # Features
    feature_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']
    optional_features = ['Vict Sex', 'Vict Descent', 'Premis Cd', 'Part 1-2']
    for col in optional_features:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    y = df['Crime_Group']

    print(f"   - Features ({len(feature_cols)}) : {feature_cols}")
    print(f"   - Shape X : {X.shape}")
    print(f"   - Distribution Crime_Group :\n{y.value_counts(normalize=True).round(3)}")

    # Encodage du label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded
    )

    print(f"\n📊 Split : Train={X_train.shape[0]} | Test={X_test.shape[0]}")

    # ---------------------------------------------------------
    # Imputation séparée : numérique (médiane) + catégorique (mode)
    # ---------------------------------------------------------
    print("\n🔧 Imputation des valeurs manquantes...")

    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    print(f"   - Colonnes numériques imputées : {numeric_cols}")
    print(f"   - Colonnes catégorielles imputées : {categorical_cols}")
    print(f"   - NaN restants : {X_train.isna().sum().sum()}")

    # ---------------------------------------------------------
    # Encodage des variables catégorielles
    # ---------------------------------------------------------
    print("\n🔠 Encodage des variables catégorielles...")
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
        print(f"   - Encodées avec OrdinalEncoder : {categorical_cols}")

    # ---------------------------------------------------------
    # SMOTE pour équilibrage
    # ---------------------------------------------------------
    print("\n⚖️ Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   - Distribution après SMOTE : {Counter(y_train_res)}")

    # ---------------------------------------------------------
    # Entraînement
    # ---------------------------------------------------------
    train_model(X_train_res, X_test, y_train_res, y_test, args.model, le, feature_cols)

    print(f"\n✅ Modèle {args.model.upper()} entraîné avec succès !")
    print("   → Voir les résultats dans MLflow : mlflow ui")


if __name__ == "__main__":
    main()