"""
Script pour charger et utiliser le modèle en production depuis MLflow (DagsHub)
Modes : demo, interactive, batch
"""

import mlflow
import dagshub
import pandas as pd
import argparse
from mlflow.tracking import MlflowClient

# =========================================================
# Classes de crimes (ordre standard – doit correspondre à l'entraînement)
# =========================================================
CRIME_CLASSES = [
    "Other / Fraud / Public Order Crime",
    "Property & Theft Crime",
    "Vehicle-Related Crime",
    "Violent Crime"
]

def connect_to_mlflow():
    """Initialise la connexion à MLflow via DagsHub"""
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    print("✅ Connecté à MLflow via DagsHub")


def list_registered_models(model_name="crime-prediction-model"):
    """Affiche les versions disponibles du modèle dans le registry"""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name)
        if not versions:
            print(f"⚠️ Aucune version trouvée pour le modèle '{model_name}'")
            return False

        print(f"\n📋 Versions disponibles pour '{model_name}' :")
        print(f"{'Version':<8} {'Stage':<12} {'Run ID':<20} {'Description'}")
        print("-" * 70)
        for v in sorted(versions, key=lambda x: x.version, reverse=True):
            desc = (v.description[:50] + '...') if v.description and len(v.description) > 50 else (v.description or "N/A")
            print(f"{v.version:<8} {v.current_stage:<12} {v.run_id[:12]:<20} {desc}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des modèles : {e}")
        return False


def load_production_model(model_name="crime-prediction-model"):
    """
    Charge le modèle actuellement en stage Production
    Retourne None si aucun modèle en Production
    """
    print(f"\n🔄 Tentative de chargement du modèle '{model_name}' en Production...")

    client = MlflowClient()
    try:
        # Vérifie s'il y a une version en Production
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            print(f"❌ Aucun modèle '{model_name}' en stage Production trouvé.")
            print("\n💡 Que faire ?")
            print("   1. Lancez : python src/models/promote_best_model.py --compare_all --auto_promote")
            print("   2. Ou promouvez manuellement un bon run depuis l'UI DagsHub MLflow")
            list_registered_models(model_name)
            return None

        version = prod_versions[0]
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"✅ Modèle chargé avec succès !")
        print(f"   • Version : {version.version}")
        print(f"   • Stage   : {version.current_stage}")
        print(f"   • Run ID  : {version.run_id}")
        print(f"   • Créé le : {pd.Timestamp(version.creation_timestamp, unit='ms')}")

        # Affichage des métriques principales
        try:
            run = client.get_run(version.run_id)
            print(f"\n📊 Métriques du modèle :")
            metrics = run.data.metrics
            important = {k: v for k, v in metrics.items() if any(x in k.lower() for x in ['test', 'cv', 'f1', 'accuracy'])}
            for k, v in important.items():
                print(f"   {k}: {v:.4f}")
        except:
            pass

        return model

    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        list_registered_models(model_name)
        return None


def predict_crime(model, features_dict):
    """Prédiction unique"""
    df = pd.DataFrame([features_dict])
    pred_code = model.predict(df)[0]
    return int(pred_code)


def get_crime_class_name(pred_code):
    """Convertit le code numérique en nom de classe"""
    try:
        return CRIME_CLASSES[int(pred_code)]
    except:
        return f"Classe inconnue ({pred_code})"


def demo_predictions():
    """Démo avec scénarios prédéfinis"""
    print("\n" + "="*80)
    print("🎯 DÉMONSTRATION DE PRÉDICTIONS")
    print("="*80)

    model = load_production_model()
    if model is None:
        return

    scenarios = [
        ("Vol en soirée dans un quartier calme", {
            'Hour': 21, 'Day_of_week': 5, 'Month_num': 8, 'LAT': 34.0615, 'LON': -118.3523, 'Vict Age': 40, 'AREA': 15
        }),
        ("Agression en centre-ville l'après-midi", {
            'Hour': 15, 'Day_of_week': 3, 'Month_num': 6, 'LAT': 34.0522, 'LON': -118.2437, 'Vict Age': 27, 'AREA': 1
        }),
        ("Vol de véhicule tôt le matin", {
            'Hour': 5, 'Day_of_week': 1, 'Month_num': 10, 'LAT': 34.0420, 'LON': -118.2630, 'Vict Age': 35, 'AREA': 9
        }),
        ("Activité suspecte tard la nuit", {
            'Hour': 2, 'Day_of_week': 6, 'Month_num': 12, 'LAT': 34.0500, 'LON': -118.2500, 'Vict Age': 22, 'AREA': 3
        })
    ]

    print("\n📍 Prédictions pour différents scénarios :\n")
    for i, (name, features) in enumerate(scenarios, 1):
        pred_code = predict_crime(model, features)
        crime_type = get_crime_class_name(pred_code)
        print(f"{i}. {name}")
        print(f"    → Heure: {features['Hour']}h | Jour: {features['Day_of_week']} | Mois: {features['Month_num']}")
        print(f"    🎯 Prédiction : {crime_type}")
        print()


def interactive_prediction():
    """Mode interactif"""
    print("\n" + "="*80)
    print("🎮 MODE INTERACTIF - PRÉDICTION DE TYPE DE CRIME")
    print("="*80)

    model = load_production_model()
    if model is None:
        return

    print("\n📝 Saisissez les informations (valeurs typiques pour Los Angeles) :\n")
    try:
        features = {
            'Hour': int(input("   Heure (0-23)                 : ")),
            'Day_of_week': int(input("   Jour (0=Lun ... 6=Dim)       : ")),
            'Month_num': int(input("   Mois (1-12)                   : ")),
            'LAT': float(input("   Latitude (ex: 34.0522)        : ")),
            'LON': float(input("   Longitude (ex: -118.2437)     : "))
        }

        vict_age = input("   Âge victime (optionnel)       : ").strip()
        if vict_age:
            features['Vict Age'] = float(vict_age)

        area = input("   Code AREA (1-21, optionnel)   : ").strip()
        if area:
            features['AREA'] = int(area)

        print("\n🔮 Analyse en cours...")
        pred_code = predict_crime(model, features)
        crime_type = get_crime_class_name(pred_code)

        print("\n" + "="*80)
        print("🎯 RÉSULTAT")
        print("="*80)
        print(f" Type de crime prédit : {crime_type}")
        print(f" Code interne         : {pred_code}")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Erreur de saisie : {e}")


def batch_prediction_from_csv(csv_path):
    """Prédictions sur un fichier CSV entier"""
    print(f"\n📂 Chargement du fichier : {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"   → {len(df)} échantillons chargés")
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    model = load_production_model()
    if model is None:
        return

    required_cols = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Colonnes manquantes : {missing}")
        return

    print("\n🔮 Prédictions en cours sur tout le fichier...")
    predictions = model.predict(df[required_cols + ['Vict Age', 'AREA']])  # inclut optionnelles si présentes

    df['Predicted_Crime_Code'] = predictions
    df['Predicted_Crime_Type'] = [get_crime_class_name(p) for p in predictions]

    output_path = csv_path.replace('.csv', '_with_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Fichier avec prédictions sauvegardé : {output_path}")

    print(f"\n📊 Répartition des prédictions :")
    print(df['Predicted_Crime_Type'].value_counts())


def main():
    parser = argparse.ArgumentParser(description='Utiliser le modèle de prédiction de crimes en production')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'interactive', 'batch'],
                        help='Mode d\'exécution')
    parser.add_argument('--csv', type=str, help='Chemin du CSV pour le mode batch')
    parser.add_argument('--model_name', type=str, default='crime-prediction-model',
                        help='Nom du modèle dans le Model Registry')

    args = parser.parse_args()

    connect_to_mlflow()

    if args.mode == 'demo':
        demo_predictions()
    elif args.mode == 'interactive':
        interactive_prediction()
    elif args.mode == 'batch':
        if not args.csv:
            print("❌ Mode batch nécessite --csv chemin/vers/fichier.csv")
            return
        batch_prediction_from_csv(args.csv)

    print("\n✅ Opération terminée.")


if __name__ == "__main__":
    main()