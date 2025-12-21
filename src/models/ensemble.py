"""
Script pour créer des modèles ensemble (Voting et Stacking)
Tracking MLflow + DagsHub – Version avec nom dynamique selon l'année
"""

import os
import argparse
import warnings
from collections import Counter

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# =========================================================
# Dossiers
# =========================================================
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# =========================================================
# Chargement ou création des modèles de base (avec année dynamique)
# =========================================================
def load_or_create_model(name, year):
    path = f"models/{name}_{year}_baseline.joblib"
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"   ✅ Modèle chargé : {name}_{year}_baseline")
    else:
        print(f"   ⚠️ Modèle {name}_{year}_baseline non trouvé → création par défaut")
        if name == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
        elif name == "xgboost":
            model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        elif name == "lightgbm":
            model = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
        else:
            raise ValueError(f"Modèle inconnu : {name}")
    return name, model


# =========================================================
# Entraînement Voting
# =========================================================
def train_voting_ensemble(X_train, X_test, y_train, y_test, le, voting_type="soft", year="2020"):
    run_name = f"voting_{voting_type}_{year}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n🗳️ Ensemble Voting ({voting_type.upper()}) - Année {year}")

        base_models = [
            load_or_create_model("random_forest", year),
            load_or_create_model("xgboost", year),
            load_or_create_model("lightgbm", year)
        ]

        clf = VotingClassifier(
            estimators=base_models,
            voting=voting_type,
            n_jobs=-1
        )

        print("   🔄 Entraînement en cours...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_params({
            "ensemble_type": "voting",
            "voting": voting_type,
            "base_models": "rf,xgb,lgbm",
            "n_classes": len(le.classes_),
            "year": year
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1_weighted": f1
        })

        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())

        # Signature et input example
        from mlflow.models.signature import infer_signature
        input_example = pd.DataFrame(X_train[:1])
        signature = infer_signature(X_train[:100], y_pred[:100])

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Sauvegarde avec nom dynamique
        model_path = f"models/voting_{voting_type}_{year}.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats Voting ({voting_type.upper()} - {year}) :")
        print(f"   ✓ Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   ✓ Test F1-Score : {f1:.4f}")
        print(f"   ✓ CV Accuracy   : {cv_scores.mean():.4f}")
        print("\n   Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        return acc


# =========================================================
# Entraînement Stacking
# =========================================================
def train_stacking_ensemble(X_train, X_test, y_train, y_test, le, year="2020"):
    run_name = f"stacking_{year}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n📚 Ensemble Stacking - Année {year}")

        base_models = [
            load_or_create_model("random_forest", year),
            load_or_create_model("xgboost", year),
            load_or_create_model("lightgbm", year)
        ]

        clf = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42, class_weight='balanced'),
            cv=5,
            n_jobs=-1,
            passthrough=True
        )

        print("   🔄 Entraînement en cours (plus long)...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_params({
            "ensemble_type": "stacking",
            "meta_learner": "logistic_regression",
            "cv_folds": 5,
            "passthrough": True,
            "base_models": "rf,xgb,lgbm",
            "n_classes": len(le.classes_),
            "year": year
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1_weighted": f1
        })

        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())

        from mlflow.models.signature import infer_signature
        input_example = pd.DataFrame(X_train[:1])
        signature = infer_signature(X_train[:100], y_pred[:100])

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Sauvegarde avec nom dynamique
        model_path = f"models/stacking_{year}.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats Stacking ({year}) :")
        print(f"   ✓ Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   ✓ Test F1-Score : {f1:.4f}")
        print(f"   ✓ CV Accuracy   : {cv_scores.mean():.4f}")
        print("\n   Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        return acc


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles ensemble")
    parser.add_argument('--data', type=str, required=True,
                        help="Chemin du fichier processed (ex: data/processed/crime_2021_processed.csv)")
    parser.add_argument("--ensemble", choices=["voting", "stacking", "both"], default="both")
    parser.add_argument("--voting", choices=["soft", "hard"], default="soft")
    args = parser.parse_args()

    # Initialisation MLflow
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        print(f"✅ MLflow Tracking URI: {tracking_uri}")
    else:
        print("⚠️ MLflow Tracking URI non défini, utilisation locale")
    mlflow.set_experiment("crime-prediction-ensemble")

    # Extraction de l'année depuis le nom du fichier
    filename = os.path.basename(args.data)
    year = filename.split('_')[-2]  # "crime_2021_processed.csv" → "2021"
    print(f"📅 Année détectée : {year}")

    print("=" * 80)
    print(f"🚀 ENTRAÎNEMENT DES MODÈLES ENSEMBLE - ANNÉE {year}")
    print("=" * 80)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Fichier introuvable : {args.data}")

    df = pd.read_csv(args.data)

    if 'Crime_Group' not in df.columns:
        raise ValueError("'Crime_Group' manquant dans les données")

    feature_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']
    optional_features = ['Vict Sex', 'Vict Descent', 'Premis Cd', 'Part 1-2']
    for col in optional_features:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    y = df['Crime_Group']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Imputation + Encodage + SMOTE
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    results = {}

    if args.ensemble in ["voting", "both"]:
        acc_voting = train_voting_ensemble(
            X_train_res, X_test, y_train_res, y_test, le, voting_type=args.voting, year=year
        )
        results[f'voting_{args.voting}'] = acc_voting

    if args.ensemble in ["stacking", "both"]:
        acc_stacking = train_stacking_ensemble(
            X_train_res, X_test, y_train_res, y_test, le, year=year
        )
        results['stacking'] = acc_stacking

    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print("\n📊 Résumé des performances :")
    for name, acc in results.items():
        print(f"   • {name}_{year} : {acc:.4f} ({acc*100:.2f}%)")

    print(f"\n💡 Modèles sauvegardés dans /models avec suffixe _{year}")
    print("   → Trackés dans MLflow (experiment: crime-prediction-ensemble)")


if __name__ == "__main__":
    main()