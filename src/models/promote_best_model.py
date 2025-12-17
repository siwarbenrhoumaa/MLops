import mlflow
import dagshub
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import argparse
import datetime
import joblib
import os
import tempfile

def connect_to_mlflow():
    """Connecte au serveur MLflow via DagsHub"""
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    client = MlflowClient()
    print("✅ Connecté à MLflow via DagsHub")
    return client

def find_joblib_artifact(client, run_id):
    """
    Recherche un fichier .joblib ou .pkl à la racine des artifacts du run.
    Retourne le chemin relatif (ex: 'stacking.joblib')
    """
    print(f"\n🔍 Recherche d'un fichier .joblib ou .pkl dans le run {run_id}...")
    artifacts = client.list_artifacts(run_id)
    
    for artifact in artifacts:
        if not artifact.is_dir and artifact.path.lower().endswith(('.joblib', '.pkl')):
            print(f"✅ Modèle trouvé : {artifact.path}")
            return artifact.path
    
    raise Exception("Aucun fichier .joblib ou .pkl trouvé à la racine des artifacts.")

def register_model_from_joblib(client, run_id, model_name="crime-prediction-model"):
    """
    Télécharge le .joblib, le charge, et l'enregistre proprement dans le Model Registry
    via un nouveau run temporaire.
    """
    print(f"\n📝 Enregistrement du modèle depuis artifact .joblib dans le Model Registry...")
    
    try:
        # 1. Trouver le fichier joblib
        joblib_path = find_joblib_artifact(client, run_id)
        
        # 2. Télécharger localement
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, joblib_path, dst_path=tmpdir)
            full_local_path = os.path.join(tmpdir, joblib_path)
            
            print(f"📥 Modèle téléchargé temporairement : {full_local_path}")
            
            # 3. Charger le modèle
            model = joblib.load(full_local_path)
            print("✅ Modèle chargé avec joblib")
            
            # 4. Créer un nouveau run pour logger proprement le modèle
            with mlflow.start_run(run_name=f"promote-fix-{run_id[:8]}"):
                # Copier les métriques et params du run original (pour traçabilité)
                original_run = client.get_run(run_id)
                for k, v in original_run.data.metrics.items():
                    mlflow.log_metric(k, v)
                for k, v in original_run.data.params.items():
                    mlflow.log_param(k, v)
                for k, v in original_run.data.tags.items():
                    mlflow.set_tag(k, v)
                
                mlflow.set_tag("original_run_id", run_id)
                mlflow.set_tag("promotion_source", "manual_joblib_fix")
                
                # Logger le modèle correctement
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name  # Enregistre directement une nouvelle version
                )
                
                new_run_id = mlflow.active_run().info.run_id
                print(f"✅ Modèle re-loggé proprement dans le nouveau run : {new_run_id}")
            
            # Récupérer la dernière version créée
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            print(f"✅ Nouvelle version enregistrée : {model_name} v{latest_version.version}")
            
            return latest_version
            
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement depuis joblib : {e}")
        return None

# === Les autres fonctions restent identiques ===
def get_best_model_from_experiment(client, experiment_name, metric='test_accuracy'):
    print(f"\n🔍 Recherche du meilleur modèle dans l'experiment : {experiment_name}")
    
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' introuvable")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric} DESC"],
        max_results=50
    )
    
    if not runs:
        print(f"❌ Aucun run trouvé dans l'experiment '{experiment_name}'")
        return None
    
    print(f"\n📊 {len(runs)} runs trouvés. Top 10 :")
    print(f"{'Rank':<6} {'Run Name':<30} {'Model':<20} {metric:<15} {'F1-Score':<15}")
    print("="*90)
    
    results = []
    for i, run in enumerate(runs[:10], 1):
        run_name = run.data.tags.get('mlflow.runName', 'N/A')
        model_type = run.data.params.get('model_type', run.data.params.get('ensemble_type', 'N/A'))
        test_acc = run.data.metrics.get(metric, 0)
        test_f1 = run.data.metrics.get('test_f1', 0)
        
        results.append({
            'rank': i,
            'run_id': run.info.run_id,
            'run_name': run_name,
            'model_type': model_type,
            'test_accuracy': test_acc,
            'test_f1': test_f1
        })
        
        print(f"{i:<6} {run_name:<30} {model_type:<20} {test_acc:<15.4f} {test_f1:<15.4f}")
    
    best_run = runs[0]
    best_metric = best_run.data.metrics.get(metric, 0)
    
    print("\n" + "="*90)
    print(f"🏆 MEILLEUR MODÈLE :")
    print(f" Run ID : {best_run.info.run_id}")
    print(f" Run Name : {best_run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f" Model Type : {best_run.data.params.get('model_type', run.data.params.get('ensemble_type', 'N/A'))}")
    print(f" {metric} : {best_metric:.4f}")
    print(f" Test F1-Score : {best_run.data.metrics.get('test_f1', 0):.4f}")
    print("="*90)
    
    return {
        'run': best_run,
        'run_id': best_run.info.run_id,
        'run_name': best_run.data.tags.get('mlflow.runName', 'N/A'),
        'model_type': best_run.data.params.get('model_type', run.data.params.get('ensemble_type', 'N/A')),
        'metric_value': best_metric,
        'results_df': pd.DataFrame(results)
    }

def transition_model_to_production(client, model_name, version, archive_existing=True):
    print(f"\n🚀 Promotion du modèle en production...")
    try:
        if archive_existing:
            print(" 📦 Archivage des anciennes versions en production...")
            current_prod_models = client.get_latest_versions(model_name, stages=["Production"])
            for model in current_prod_models:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model.version,
                    stage="Archived"
                )
                print(f" ✓ Version {model.version} archivée")
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_existing
        )
        
        print(f"✅ Modèle {model_name} v{version} est maintenant en PRODUCTION")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la promotion : {e}")
        return False

def add_model_description(client, model_name, version, description):
    try:
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print(f"✅ Description ajoutée au modèle")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter la description : {e}")

def get_production_model_info(client, model_name):
    print(f"\n🔍 Informations sur le modèle en production...")
    try:
        prod_models = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_models:
            print(f"❌ Aucun modèle en production pour '{model_name}'")
            return None
        
        model = prod_models[0]
        print(f"\n✅ Modèle en Production :")
        print(f" Nom : {model.name}")
        print(f" Version : {model.version}")
        print(f" Run ID : {model.run_id}")
        print(f" Description : {model.description or 'N/A'}")
        
        run = client.get_run(model.run_id)
        print(f"\n 📊 Métriques :")
        for metric_name, metric_value in sorted(run.data.metrics.items()):
            print(f" {metric_name}: {metric_value:.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None

def compare_experiments(client):
    print("\n📊 COMPARAISON DES EXPERIMENTS\n")
    print("="*100)
    
    experiments = ['crime-prediction-baseline', 'crime-prediction-ensemble']
    all_results = []
    
    for exp_name in experiments:
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            continue
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_accuracy DESC"],
            max_results=3
        )
        
        for run in runs:
            run_name = run.data.tags.get('mlflow.runName', 'N/A')
            model_type = run.data.params.get('model_type', run.data.params.get('ensemble_type', 'N/A'))
            
            all_results.append({
                'Experiment': exp_name,
                'Run Name': run_name,
                'Model/Ensemble': model_type,
                'Test Accuracy': run.data.metrics.get('test_accuracy', 0),
                'Test F1': run.data.metrics.get('test_f1', 0),
                'CV Mean': run.data.metrics.get('cv_mean_accuracy', 0),
                'Run ID': run.info.run_id
            })
    
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values('Test Accuracy', ascending=False)
        print(df.to_string(index=False))
        print("="*100)
        return df
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Sélectionner et promouvoir le meilleur modèle en production')
    parser.add_argument('--experiment', type=str, default='crime-prediction-ensemble',
                        help='Nom de l\'experiment')
    parser.add_argument('--model_name', type=str, default='crime-prediction-model',
                        help='Nom du modèle dans le registry')
    parser.add_argument('--metric', type=str, default='test_accuracy',
                        choices=['test_accuracy', 'test_f1', 'cv_mean_accuracy'],
                        help='Métrique pour sélectionner le meilleur modèle')
    parser.add_argument('--auto_promote', action='store_true',
                        help='Promouvoir automatiquement sans confirmation')
    parser.add_argument('--compare_all', action='store_true',
                        help='Comparer tous les experiments avant de choisir')
    
    args = parser.parse_args()
    
    client = connect_to_mlflow()
    
    if args.compare_all:
        compare_experiments(client)
    
    best_model = get_best_model_from_experiment(client, args.experiment, args.metric)
    
    if best_model is None:
        print("\n❌ Impossible de continuer sans modèle sélectionné")
        return
    
    if not args.auto_promote:
        print(f"\n⚠️ Voulez-vous promouvoir ce modèle en production ?")
        print(f" Run : {best_model['run_name']}")
        print(f" Run ID : {best_model['run_id']}")
        print(f" Metric : {args.metric} = {best_model['metric_value']:.4f}")
        
        response = input("\n Confirmer (o/n) ? : ").strip().lower()
        if response != 'o':
            print("❌ Opération annulée")
            return
    
    # Nouvelle fonction qui gère le cas joblib
    model_version = register_model_from_joblib(client, best_model['run_id'], args.model_name)
    
    if model_version is None:
        print("\n❌ Échec de l'enregistrement du modèle")
        return
    
    # Description enrichie
    description = f"""
    Modèle promu automatiquement depuis un artifact .joblib (fix manuel).
    
    Run original : {best_model['run_name']} ({best_model['run_id']})
    Type : {best_model['model_type']}
    Métrique sélection : {args.metric} = {best_model['metric_value']:.4f}
    Test F1 : {best_model['run'].data.metrics.get('test_f1', 0):.4f}
    
    Date promotion : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    add_model_description(client, args.model_name, model_version.version, description)
    
    success = transition_model_to_production(client, args.model_name, model_version.version, archive_existing=True)
    
    if success:
        get_production_model_info(client, args.model_name)
        print("\n" + "="*90)
        print("🎉 PROCESSUS TERMINÉ AVEC SUCCÈS !")
        print("="*90)
        print(f"\n💡 Pour charger le modèle en production :")
        print(f"   model = mlflow.pyfunc.load_model('models:/{args.model_name}/Production')")
    else:
        print("\n❌ Échec de la promotion en production")

if __name__ == "__main__":
    main()