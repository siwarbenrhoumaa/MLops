"""
Script pour nettoyer les runs MLflow problématiques
"""

import mlflow
import dagshub
from mlflow.tracking import MlflowClient

def clean_problematic_runs():
    """Supprime les runs sans modèle loggé"""
    
    # Connexion
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    client = MlflowClient()
    
    print("=" * 80)
    print("🧹 NETTOYAGE DES RUNS PROBLÉMATIQUES")
    print("=" * 80)
    
    # Récupérer l'experiment ensemble
    experiment = client.get_experiment_by_name("crime-prediction-ensemble")
    
    if experiment is None:
        print("❌ Experiment 'crime-prediction-ensemble' introuvable")
        return
    
    print(f"\n📊 Experiment trouvé : {experiment.name} (ID: {experiment.experiment_id})")
    
    # Lister tous les runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    print(f"\n📋 {len(runs)} runs trouvés\n")
    
    problematic_runs = []
    
    for run in runs:
        run_id = run.info.run_id
        run_name = run.data.tags.get('mlflow.runName', 'N/A')
        
        # Vérifier si le run a un modèle loggé
        try:
            artifacts = client.list_artifacts(run_id)
            has_model = any(art.path == "model" for art in artifacts)
            
            if not has_model:
                problematic_runs.append({
                    'run_id': run_id,
                    'run_name': run_name,
                    'start_time': run.info.start_time
                })
                print(f"⚠️  Run sans modèle : {run_name} (ID: {run_id[:8]}...)")
        except Exception as e:
            print(f"❌ Erreur vérification {run_name} : {e}")
    
    if not problematic_runs:
        print("✅ Aucun run problématique trouvé !")
        return
    
    print(f"\n📊 Total: {len(problematic_runs)} runs sans modèle")
    
    # Demander confirmation
    print("\n⚠️  Voulez-vous supprimer ces runs ? (o/n) : ", end="")
    response = input().strip().lower()
    
    if response != 'o':
        print("❌ Opération annulée")
        return
    
    # Supprimer les runs
    print("\n🗑️  Suppression en cours...")
    deleted = 0
    
    for run in problematic_runs:
        try:
            client.delete_run(run['run_id'])
            print(f"   ✓ Supprimé : {run['run_name']}")
            deleted += 1
        except Exception as e:
            print(f"   ✗ Erreur : {run['run_name']} - {e}")
    
    print(f"\n✅ {deleted}/{len(problematic_runs)} runs supprimés")
    print("\n💡 Vous pouvez maintenant ré-entraîner les ensembles :")
    print("   python src/models/ensemble.py --ensemble both")


if __name__ == "__main__":
    clean_problematic_runs()
