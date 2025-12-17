"""
Script pour créer des modèles ensemble (Voting et Stacking)
Tracking MLflow + DagsHub – Version CORRIGÉE
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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
# Mapping des crimes
# =========================================================
def map_crime_group_4(desc):
    if pd.isna(desc):
        return "Other / Fraud / Public Order Crime"
    desc = str(desc).upper()

    if any(k in desc for k in ["ASSAULT", "BATTERY", "ROBBERY", "HOMICIDE", "RAPE", "SEX", "SODOMY", "LEWD"]):
        return "Violent Crime"
    if any(k in desc for k in ["THEFT", "BURGLARY", "SHOPLIFTING", "VANDALISM", "ARSON"]):
        return "Property & Theft Crime"
    if any(k in desc for k in ["VEHICLE", "DWOC", "MOTOR VEHICLE", "BOAT"]):
        return "Vehicle-Related Crime"
    return "Other / Fraud / Public Order Crime"


# =========================================================
# Création des modèles de base
# =========================================================
def create_default_model(name):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    raise ValueError(f"Modèle inconnu : {name}")


def load_or_create_model(name):
    path = f"models/{name}_baseline.joblib"
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"   ✅ Chargé : {name}")
    else:
        model = create_default_model(name)
        print(f"   ⚠️ Créé (non trouvé) : {name}")
    return name, model


# =========================================================
# Entraînement Voting
# =========================================================
def train_voting_ensemble(X_train, X_test, y_train, y_test, le, voting_type="soft"):
    run_name = f"voting_{voting_type}"
    
    # ✅ PAS DE nested=True - run indépendant
    with mlflow.start_run(run_name=run_name):
        print(f"\n🗳️ Ensemble Voting ({voting_type})")

        # Chargement ou création des modèles de base
        base_models = [
            load_or_create_model("random_forest"),
            load_or_create_model("xgboost"),
            load_or_create_model("lightgbm")
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

        # Logging MLflow
        mlflow.log_params({
            "ensemble_type": "voting",
            "voting": voting_type,
            "base_models": "rf,xgb,lgbm",
            "n_classes": len(le.classes_)
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1": f1
        })

        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())

        # ✅ Logging du modèle avec signature et input_example
        from mlflow.models.signature import infer_signature
        
        # Créer un input_example (DataFrame)
        input_example = pd.DataFrame(
            X_train[:1],
            columns=['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']
        )
        
        # Inférer la signature
        predictions = clf.predict(X_train[:100])
        signature = infer_signature(X_train[:100], predictions)
        
        # Logger le modèle CORRECTEMENT
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Sauvegarde locale
        model_path = f"models/voting_{voting_type}.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats Voting ({voting_type}) :")
        print(f"   ✓ Test Accuracy : {acc:.4f}")
        print(f"   ✓ Test F1-Score : {f1:.4f}")
        print(f"   ✓ CV Accuracy   : {cv_scores.mean():.4f}")
        print(f"\n   Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        return acc


# =========================================================
# Entraînement Stacking
# =========================================================
def train_stacking_ensemble(X_train, X_test, y_train, y_test, le):
    run_name = "stacking"
    
    # ✅ PAS DE nested=True - run indépendant
    with mlflow.start_run(run_name=run_name):
        print(f"\n📚 Ensemble Stacking (meta-learner: LogisticRegression)")

        base_models = [
            load_or_create_model("random_forest"),
            load_or_create_model("xgboost"),
            load_or_create_model("lightgbm")
        ]

        clf = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(
                max_iter=1000, 
                n_jobs=-1, 
                random_state=42,
                class_weight='balanced'
            ),
            cv=5,
            n_jobs=-1,
            passthrough=True
        )

        print("   🔄 Entraînement en cours (cela peut prendre quelques minutes)...")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Logging MLflow
        mlflow.log_params({
            "ensemble_type": "stacking",
            "meta_learner": "logistic_regression",
            "cv_folds": 5,
            "passthrough": True,
            "base_models": "rf,xgb,lgbm",
            "n_classes": len(le.classes_)
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1": f1
        })

        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())

        # ✅ Logging du modèle avec signature et input_example
        from mlflow.models.signature import infer_signature
        
        input_example = pd.DataFrame(
            X_train[:1],
            columns=['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']
        )
        
        predictions = clf.predict(X_train[:100])
        signature = infer_signature(X_train[:100], predictions)
        
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Sauvegarde locale
        model_path = "models/stacking.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats Stacking :")
        print(f"   ✓ Test Accuracy : {acc:.4f}")
        print(f"   ✓ Test F1-Score : {f1:.4f}")
        print(f"   ✓ CV Accuracy   : {cv_scores.mean():.4f}")
        print(f"\n   Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        return acc


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles ensemble")
    parser.add_argument("--ensemble", choices=["voting", "stacking", "both"], default="both",
                        help="Type d'ensemble à entraîner")
    parser.add_argument("--voting", choices=["soft", "hard"], default="soft",
                        help="Type de voting (si voting sélectionné)")
    args = parser.parse_args()

    # Initialisation DagsHub + MLflow
    dagshub.init(repo_owner="benrhoumamohamed752", repo_name="ProjetMLOps", mlflow=True)
    mlflow.set_experiment("crime-prediction-ensemble")

    print("=" * 80)
    print("🚀 ENTRAÎNEMENT DES MODÈLES ENSEMBLE")
    print("=" * 80)
    
    print("\n📂 Chargement des données...")
    df = pd.read_csv("data/processed/crime_2020_processed.csv")
    df["Crime_Group"] = df["Crm Cd Desc"].apply(map_crime_group_4)

    features = ["Hour", "Day_of_week", "Month_num", "LAT", "LON", "Vict Age", "AREA"]
    X = df[features]
    y_raw = df["Crime_Group"]

    # LabelEncoder partagé (important pour cohérence)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"\n📊 Classes détectées : {list(le.classes_)}")
    print(f"   Distribution initiale :")
    for class_name in le.classes_:
        count = (y_raw == class_name).sum()
        print(f"      {class_name}: {count}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📊 Split des données :")
    print(f"   Train: {X_train.shape[0]} échantillons")
    print(f"   Test : {X_test.shape[0]} échantillons")

    # Imputation
    print("\n🔧 Imputation des valeurs manquantes...")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # SMOTE
    print("\n⚖️ Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   Distribution après SMOTE :")
    for i, class_name in enumerate(le.classes_):
        count = (y_train_res == i).sum()
        print(f"      {class_name}: {count}")

    # Entraînement
    print("\n" + "=" * 80)
    print("📈 DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 80)
    
    results = {}
    
    if args.ensemble in ["voting", "both"]:
        acc_voting = train_voting_ensemble(
            X_train_res, X_test, y_train_res, y_test, le, voting_type=args.voting
        )
        results[f'voting_{args.voting}'] = acc_voting

    if args.ensemble in ["stacking", "both"]:
        acc_stacking = train_stacking_ensemble(
            X_train_res, X_test, y_train_res, y_test, le
        )
        results['stacking'] = acc_stacking

    # Résumé final
    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    
    print("\n📊 Résumé des performances :")
    for model_name, accuracy in results.items():
        print(f"   • {model_name:20} : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n💡 Prochaines étapes :")
    print("   1. Comparer avec les modèles baseline :")
    print("      python src/models/promote_best_model.py --compare_all")
    print()
    print("   2. Promouvoir le meilleur modèle :")
    print("      python src/models/promote_best_model.py --experiment crime-prediction-baseline --auto_promote")
    print()
    print("   3. Tester le modèle en production :")
    print("      python src/models/use_production_model.py --mode demo")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
