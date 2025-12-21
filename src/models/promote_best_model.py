"""
Script de comparaison et promotion du meilleur modèle parmi ceux d'une même année
Utilise le paramètre 'year' loggé dans MLflow pour filtrer
"""

import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import argparse
import os

def connect_to_mlflow():
    # Les env vars sont déjà configurées par le workflow
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        print(f"✅ MLflow Tracking URI: {tracking_uri}")
    
    client = MlflowClient()
    return client

def get_models_by_year(client, target_year):
    """
    Récupère tous les runs contenant le paramètre 'year' = target_year
    """
    print("=" * 130)
    print(f"📊 COMPARAISON DES MODÈLES POUR L'ANNÉE {target_year}")
    print("=" * 130)

    # Tous les experiments possibles
    experiment_names = [
        'crime-prediction-baseline',
        'crime-prediction-ensemble'
    ]

    all_results = []

    for exp_name in experiment_names:
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            continue

        print(f"\n🔍 Analyse de l'experiment : {exp_name}")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            filter_string=f"params.year = '{target_year}'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=100
        )

        print(f"   → {len(runs)} runs trouvés pour l'année {target_year}")

        for run in runs:
            run_name = run.data.tags.get('mlflow.runName', 'N/A')
            model_type = run.data.params.get('model_type',
                                            run.data.params.get('ensemble_type', 'N/A'))

            test_acc = run.data.metrics.get('test_accuracy', 0)
            test_f1 = run.data.metrics.get('test_f1_weighted',
                                          run.data.metrics.get('test_f1', 0))
            cv_mean = run.data.metrics.get('cv_accuracy_mean', 0)
            cv_std = run.data.metrics.get('cv_accuracy_std', 0)
            train_acc = run.data.metrics.get('train_accuracy', 0)

            overfitting_gap = train_acc - test_acc if train_acc > 0 else 0

            all_results.append({
                'Type': 'Ensemble' if 'ensemble' in exp_name else 'Baseline',
                'Run Name': run_name,
                'Model': model_type,
                'Test Accuracy': test_acc,
                'Test F1': test_f1,
                'CV Mean': cv_mean,
                'CV Std': cv_std,
                'Train Acc': train_acc,
                'Overfitting Gap': overfitting_gap,
                'Run ID': run.info.run_id,
                'Created': run.info.start_time
            })

    if not all_results:
        print(f"\n❌ Aucun modèle trouvé pour l'année {target_year} !")
        return None

    df = pd.DataFrame(all_results)
    df = df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    return df

# === Les fonctions d'affichage restent IDENTIQUES ===
# (display_comparison, display_top_3, display_best_model_details, display_statistics, recommend_action)
# → Je les garde telles quelles pour conserver la structure

def display_comparison(df):
    print("\n" + "=" * 130)
    print("🏆 CLASSEMENT DES MODÈLES DE CETTE ANNÉE")
    print("=" * 130)
    
    print(f"\n{'Rank':<5} {'Type':<10} {'Model':<20} {'Run Name':<35} {'Test Acc':<12} {'Test F1':<10} {'CV Mean':<10} {'Overfit':<10}")
    print("-" * 130)
    
    for idx, row in df.iterrows():
        rank = idx + 1
        symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        overfit_symbol = "⚠️" if row['Overfitting Gap'] > 0.05 else "✅"
        
        print(f"{symbol} {rank:<3} {row['Type']:<10} {row['Model']:<20} {row['Run Name'][:34]:<35} "
              f"{row['Test Accuracy']:<12.4f} {row['Test F1']:<10.4f} {row['CV Mean']:<10.4f} "
              f"{overfit_symbol} {row['Overfitting Gap']:<9.4f}")
    
    print("-" * 130)

def display_top_3(df):
    print("\n" + "=" * 130)
    print("🏅 PODIUM - TOP 3 MODÈLES")
    print("=" * 130)
    
    medals = ["🥇", "🥈", "🥉"]
    positions = ["1er", "2ème", "3ème"]
    
    for i in range(min(3, len(df))):
        model = df.iloc[i]
        print(f"\n{medals[i]} {positions[i]} Place - {model['Model'].upper()}")
        print(f"   Type           : {model['Type']}")
        print(f"   Run Name       : {model['Run Name']}")
        print(f"   Test Accuracy  : {model['Test Accuracy']:.4f} ({model['Test Accuracy']*100:.2f}%)")
        print(f"   Test F1        : {model['Test F1']:.4f}")
        print(f"   CV Mean        : {model['CV Mean']:.4f}")
        print(f"   Stabilité      : {'✅ Stable' if model['Overfitting Gap'] < 0.05 else '⚠️ Overfit'}")

def display_best_model_details(client, df):
    best = df.iloc[0]
    
    print("\n" + "=" * 130)
    print("🏆 MEILLEUR MODÈLE DE CETTE ANNÉE - DÉTAILS")
    print("=" * 130)
    
    print(f"\n🎯 Informations Générales :")
    print(f"   • Rang             : #1 sur {len(df)} modèles")
    print(f"   • Type             : {best['Type']}")
    print(f"   • Modèle           : {best['Model']}")
    print(f"   • Run Name         : {best['Run Name']}")
    print(f"   • Run ID           : {best['Run ID']}")
    
    print(f"\n📊 Métriques de Performance :")
    print(f"   • Test Accuracy    : {best['Test Accuracy']:.4f} ({best['Test Accuracy']*100:.2f}%)")
    print(f"   • Test F1-Score    : {best['Test F1']:.4f}")
    print(f"   • CV Mean          : {best['CV Mean']:.4f}")
    print(f"   • Train Accuracy   : {best['Train Acc']:.4f}")
    
    gap = best['Overfitting Gap']
    status = "⚠️ OVERFITTING" if gap > 0.1 else "⚠️ LÉGER OVERFIT" if gap > 0.05 else "✅ STABLE"
    print(f"\n⚖️ Stabilité : {status} (gap = {gap:.4f})")

def display_statistics(df):
    print("\n" + "=" * 130)
    print("📈 STATISTIQUES DE CETTE ANNÉE")
    print("=" * 130)
    
    print("\n📊 Moyennes par Type :")
    print(df.groupby('Type')['Test Accuracy'].agg(['mean', 'std', 'min', 'max', 'count']).round(4))
    
    print("\n📊 Moyennes par Modèle :")
    print(df.groupby('Model')['Test Accuracy'].agg(['mean', 'count']).round(4))

def recommend_action(df, year):
    best = df.iloc[0]
    print("\n" + "=" * 130)
    print("💡 RECOMMANDATION")
    print("=" * 130)
    print(f"\n🎯 Promouvoir : {best['Model'].upper()} ({best['Run Name']})")
    print(f"   Accuracy : {best['Test Accuracy']*100:.2f}%")

# === La fonction promote_best_model reste IDENTIQUE ===
# (je la garde telle quelle, elle fonctionne parfaitement)

def promote_best_model(client, best_run_info, model_name="crime-prediction-model", auto=False):
    """
    Promouvoir automatiquement le meilleur modèle en production
    """
    import tempfile
    import os
    import joblib
    from mlflow.models.signature import infer_signature
    
    run_id = best_run_info['Run ID']
    run_name = best_run_info['Run Name']
    model_type = best_run_info['Model']
    accuracy = best_run_info['Test Accuracy']
    
    print("\n" + "=" * 130)
    print("🚀 PROMOTION EN PRODUCTION")
    print("=" * 130)
    
    print(f"\n🎯 Modèle sélectionné :")
    print(f"   • Nom           : {model_type.upper()}")
    print(f"   • Run Name      : {run_name}")
    print(f"   • Accuracy      : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Run ID        : {run_id}")
    
    # Confirmation si pas auto
    if not auto:
        print(f"\n⚠️  Voulez-vous promouvoir ce modèle en production ?")
        confirm = input("   Confirmer (o/n) ? : ").strip().lower()
        if confirm != 'o':
            print("❌ Promotion annulée")
            return False
    
    try:
        # 1. Trouver le fichier .joblib
        print(f"\n📥 Étape 1/4 : Recherche du modèle...")
        artifacts = client.list_artifacts(run_id)
        
        joblib_files = [art.path for art in artifacts if art.path.endswith('.joblib')]
        
        if not joblib_files:
            print("❌ Aucun fichier .joblib trouvé")
            return False
        
        # Prioriser certains fichiers
        priority = ['stacking', 'voting', 'ensemble', 'baseline', 'artifacts']
        joblib_path = None
        for p in priority:
            for path in joblib_files:
                if p in path.lower():
                    joblib_path = path
                    break
            if joblib_path:
                break
        
        if not joblib_path:
            joblib_path = joblib_files[0]
        
        print(f"   ✅ Modèle trouvé : {joblib_path}")
        
        # 2. Télécharger et charger le modèle
        print(f"\n📥 Étape 2/4 : Chargement du modèle...")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, joblib_path, dst_path=tmpdir)
            full_path = os.path.join(tmpdir, joblib_path)
            
            # Charger le modèle
            if full_path.endswith('_artifacts.joblib'):
                artifacts_bundle = joblib.load(full_path)
                model = artifacts_bundle.get('model', artifacts_bundle)
            else:
                model = joblib.load(full_path)
            
            print(f"   ✅ Modèle chargé : {type(model).__name__}")
            
            # 3. Créer input example et signature
            print(f"\n🔧 Étape 3/4 : Préparation de la signature...")
            
            # Récupérer les features du run original
            original_run = client.get_run(run_id)
            features_param = original_run.data.params.get('features', '')
            
            # Créer un input example
            dummy_input = pd.DataFrame({
                'Hour': [12],
                'Day_of_week': [3],
                'Month_num': [6],
                'LAT': [34.05],
                'LON': [-118.25],
                'Vict Age': [35.0],
                'AREA': [15]
            })
            
            # Ajouter colonnes optionnelles si présentes
            if 'Vict Sex' in features_param:
                dummy_input['Vict Sex'] = [0]
            if 'Vict Descent' in features_param:
                dummy_input['Vict Descent'] = [0]
            if 'Premis Cd' in features_param:
                dummy_input['Premis Cd'] = [101.0]
            if 'Part 1-2' in features_param:
                dummy_input['Part 1-2'] = [1]
            
            # Inférer la signature
            predictions = model.predict(dummy_input)
            signature = infer_signature(dummy_input, predictions)
            
            print(f"   ✅ Signature créée")
            
            # 4. Enregistrer dans MLflow
            print(f"\n📝 Étape 4/4 : Enregistrement dans Model Registry...")
            
            with mlflow.start_run(run_name=f"promote_best_{run_id[:8]}"):
                # Copier les métriques importantes
                for metric in ['test_accuracy', 'test_f1_weighted', 'cv_accuracy_mean']:
                    value = original_run.data.metrics.get(metric)
                    if value is not None:
                        mlflow.log_metric(metric, value)
                
                # Copier les paramètres
                for k, v in original_run.data.params.items():
                    mlflow.log_param(k, v)
                
                # Tags
                mlflow.set_tag("original_run_id", run_id)
                mlflow.set_tag("promoted_at", pd.Timestamp.now().isoformat())
                mlflow.set_tag("promotion_method", "auto_best_model")
                
                # Enregistrer le modèle
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=dummy_input,
                    registered_model_name=model_name
                )
            
            print(f"   ✅ Modèle enregistré dans le Model Registry")
        
        # 5. Récupérer la nouvelle version
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            print("❌ Impossible de récupérer la version créée")
            return False
        
        new_version = latest_versions[0].version
        print(f"   → Nouvelle version : v{new_version}")
        
        # 6. Archiver les anciennes versions en production
        print(f"\n📦 Archivage des anciennes versions...")
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for v in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
            print(f"   ✓ Version {v.version} archivée")
        
        # 7. Promouvoir en production
        print(f"\n🚀 Promotion en Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production"
        )
        
        # 8. Ajouter une description
        description = f"""
🏆 Meilleur modèle sélectionné automatiquement parmi 6 modèles

📊 Métriques :
   • Test Accuracy : {best_run_info['Test Accuracy']:.4f} ({best_run_info['Test Accuracy']*100:.2f}%)
   • Test F1-Score : {best_run_info['Test F1']:.4f}
   • CV Mean       : {best_run_info['CV Mean']:.4f}

🔧 Configuration :
   • Type          : {best_run_info['Type']}
   • Model         : {best_run_info['Model']}
   • Run Name      : {best_run_info['Run Name']}

📅 Promotion :
   • Date          : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
   • Run ID        : {run_id}
   • Méthode       : Comparaison automatique des 6 modèles
        """.strip()
        
        client.update_model_version(
            name=model_name,
            version=new_version,
            description=description
        )
        
        print(f"\n" + "=" * 130)
        print("✅ PROMOTION RÉUSSIE !")
        print("=" * 130)
        print(f"\n🎉 {model_name} v{new_version} est maintenant en PRODUCTION")
        print(f"\n📍 Vérifiez sur DagsHub :")
        print(f"   https://dagshub.com/benrhoumamohamed752/ProjetMLOps")
        print(f"\n💡 Charger le modèle en production :")
        print(f"   import mlflow")
        print(f"   import dagshub")
        print(f"   dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)")
        print(f"   model = mlflow.pyfunc.load_model('models:/{model_name}/Production')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la promotion : {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Comparer et promouvoir le meilleur modèle d\'une année')
    parser.add_argument('--year', type=str, required=True, help='Année à analyser (ex: 2021)')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--top', type=int, default=None)
    parser.add_argument('--promote', action='store_true')
    parser.add_argument('--auto_promote', action='store_true')
    parser.add_argument('--model_name', type=str, default='crime-prediction-model')
    args = parser.parse_args()

    client = connect_to_mlflow()

    df = get_models_by_year(client, args.year)

    if df is None:
        return

    if args.top:
        df = df.head(args.top)

    display_comparison(df)
    display_top_3(df)
    display_best_model_details(client, df)
    display_statistics(df)

    if args.save:
        os.makedirs('reports', exist_ok=True)
        df.to_csv(f'reports/models_comparison_{args.year}.csv', index=False)
        print(f"\n💾 Rapport sauvegardé : reports/models_comparison_{args.year}.csv")

    recommend_action(df, args.year)

    if args.promote or args.auto_promote:
        best_model_info = df.iloc[0].to_dict()
        success = promote_best_model(
            client,
            best_model_info,
            model_name=args.model_name,
            auto=args.auto_promote
        )
        if success:
            print(f"\nModèle {args.year} promu en Production !")

    print("\n" + "=" * 130)
    print("✅ COMPARAISON TERMINÉE")
    print("=" * 130)

if __name__ == "__main__":
    main()