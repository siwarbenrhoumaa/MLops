import argparse
import subprocess
import sys

def run_command(cmd):
    """Exécute une commande et arrête si elle échoue"""
    print(f"🚀 Exécution : {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Échec de la commande ci-dessus !")
        sys.exit(1)
    print("✅ Succès\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test complet du pipeline MLOps pour une année donnée")
    parser.add_argument('--year', type=int, required=True, help="Année du nouveau dataset à tester (ex: 2021)")
    args = parser.parse_args()

    year = args.year
    old_year = 2020  # Année de référence (ton modèle actuel)

    print(f"\n🧪 TEST DU PIPELINE MLOPS POUR L'ANNÉE {year}\n")
    print("=" * 80)

    # 1. Preprocessing du nouveau dataset
    run_command(f"python src/data/preprocessing.py --year {year}")

    # 2. Détection de drift entre old_year et new_year
    run_command(f"python src/monitoring/drift_detection.py --old_year {old_year} --new_year {year}")

    # Note : on suppose que le drift est détecté (sinon le pipeline s'arrête ici en vrai prod)
    print(f"⚠️  On continue le test en supposant que du drift a été détecté pour l'année {year}\n")

    # 3. Retraining des modèles baseline
    data_path = f"data/processed/crime_{year}_processed.csv"
    models = ["random_forest", "xgboost", "lightgbm", "logistic_regression"]

    for model_name in models:
        run_command(f"python src/models/train.py --data {data_path} --model {model_name}")

    # 4. Entraînement des ensembles
    run_command(f"python src/models/ensemble.py --data {data_path} --ensemble both")

    # 5. Promotion automatique du meilleur modèle en Production
    run_command(f"python src/models/promote_best_model.py --year {year} --auto_promote")

    print("=" * 80)
    print(f"🎉 TEST PIPELINE TERMINÉ AVEC SUCCÈS POUR L'ANNÉE {year} !")
    print("\nProchaines étapes :")
    print("   • Relancez l'API : uvicorn src.deployment.api:app --reload")
    print("   • Relancez Streamlit : streamlit run src/deployment/streamlit_app.py")
    print("   • Vérifiez la nouvelle version du modèle sur http://localhost:8000/model-info")
    print("   • Faites des prédictions pour voir le nouveau modèle en action ! 🚀")