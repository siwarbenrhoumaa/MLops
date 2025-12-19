"""
Script pour créer des modèles ensemble (Voting et Stacking)
Tracking MLflow + DagsHub – Version corrigée pour données préprocessées 2020
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
# Chargement ou création des modèles de base
# =========================================================
def load_or_create_model(name):
    # Chemin du modèle baseline entraîné avec train.py
    path = f"models/{name}_2020_baseline.joblib"
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"   ✅ Modèle chargé : {name}")
    else:
        print(f"   ⚠️ Modèle {name} non trouvé → création d'un modèle par défaut")
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
def train_voting_ensemble(X_train, X_test, y_train, y_test, le, voting_type="soft"):
    run_name = f"voting_{voting_type}_2020"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n🗳️ Ensemble Voting ({voting_type.upper()})")

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

        mlflow.log_params({
            "ensemble_type": "voting",
            "voting": voting_type,
            "base_models": "rf,xgb,lgbm",
            "n_classes": len(le.classes_)
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1_weighted": f1
        })

        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())

        # Input example et signature
        from mlflow.models.signature import infer_signature
        input_example = pd.DataFrame(X_train[:1])
        signature = infer_signature(X_train[:100], y_pred[:100])

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        model_path = f"models/voting_{voting_type}_2020.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats Voting ({voting_type.upper()}) :")
        print(f"   ✓ Test Accuracy : {acc:.4f}")
        print(f"   ✓ Test F1-Score : {f1:.4f}")
        print(f"   ✓ CV Accuracy   : {cv_scores.mean():.4f}")
        print("\n   Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        return acc


# =========================================================
# Entraînement Stacking
# =========================================================
def train_stacking_ensemble(X_train, X_test, y_train, y_test, le):
    run_name = "stacking_2020"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n📚 Ensemble Stacking (meta-learner: LogisticRegression)")

        base_models = [
            load_or_create_model("random_forest"),
            load_or_create_model("xgboost"),
            load_or_create_model("lightgbm")
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
            "n_classes": len(le.classes_)
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

        model_path = "models/stacking_2020.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path)

        print(f"\n📊 Résultats Stacking :")
        print(f"   ✓ Test Accuracy : {acc:.4f}")
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
    parser.add_argument('--data', type=str, default='data/processed/crime_2020_processed2.csv')
    parser.add_argument("--ensemble", choices=["voting", "stacking", "both"], default="both",
                        help="Type d'ensemble à entraîner")
    parser.add_argument("--voting", choices=["soft", "hard"], default="soft",
                        help="Type de voting (si voting sélectionné)")
    args = parser.parse_args()

    dagshub.init(repo_owner="benrhoumamohamed752", repo_name="ProjetMLOps", mlflow=True)
    mlflow.set_experiment("crime-prediction-ensemble-2020")

    print("=" * 80)
    print("🚀 ENTRAÎNEMENT DES MODÈLES ENSEMBLE (2020)")
    print("=" * 80)

    data_path = "data/processed/crime_2020_processed2.csv"
    print(f"\n📂 Chargement des données depuis : {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ Fichier non trouvé ! Relance preprocessed.py")
        return

    df = pd.read_csv(data_path)

    if 'Crime_Group' not in df.columns:
        print("❌ 'Crime_Group' manquant dans les données.")
        return

    # Features (identiques à train.py corrigé)
    feature_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']
    optional_features = ['Vict Sex', 'Vict Descent', 'Premis Cd', 'Part 1-2']
    for col in optional_features:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    y = df['Crime_Group']

    print(f"   - Features utilisées ({len(feature_cols)}) : {feature_cols}")
    print(f"   - Shape : {X.shape}")
    print(f"   - Distribution Crime_Group :\n{y.value_counts(normalize=True).round(3)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n📊 Split : Train={X_train.shape[0]} | Test={X_test.shape[0]}")

    # Imputation séparée
    print("\n🔧 Imputation des valeurs manquantes...")
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

    # Encodage catégorique
    print("\n🔠 Encodage des variables catégorielles...")
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder()
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # SMOTE
    print("\n⚖️ Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   - Distribution après SMOTE : {Counter(y_train_res)}")

    # Entraînement
    print("\n" + "=" * 80)
    print("📈 DÉBUT DE L'ENTRAÎNEMENT ENSEMBLE")
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

    # Résumé
    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print("\n📊 Résumé des performances :")
    for name, acc in results.items():
        print(f"   • {name:20} : {acc:.4f} ({acc*100:.2f}%)")

    print("\n💡 Tous les modèles sont sauvegardés dans /models et trackés dans MLflow !")


if __name__ == "__main__":
    main()