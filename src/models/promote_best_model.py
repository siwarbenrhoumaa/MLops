"""
Script de promotion FORCÉE du meilleur modèle de l'année
TOUJOURS remplace le modèle en Production par le meilleur de l'année actuelle
VERSION CORRIGÉE : Récupération dynamique des features depuis MLflow
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import argparse
import os
import tempfile
import joblib
from mlflow.models.signature import infer_signature


def connect_to_mlflow():
    """Connecte à MLflow via variables d'environnement"""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        print(f"✅ MLflow Tracking URI: {tracking_uri}")
    
    client = MlflowClient()
    return client


def get_production_model_info(client, model_name="crime-prediction-model"):
    """Récupère les infos du modèle actuellement en Production"""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not versions:
            print("ℹ️  Aucun modèle en Production")
            return None
        
        v = versions[0]
        run = client.get_run(v.run_id)
        
        prod_info = {
            'version': v.version,
            'run_id': v.run_id,
            'test_accuracy': run.data.metrics.get('test_accuracy', 0),
            'test_f1': run.data.metrics.get('test_f1_weighted', 0),
            'model_type': run.data.params.get('model_type') or run.data.params.get('ensemble_type', 'Unknown'),
            'year': run.data.params.get('year', 'Unknown')
        }
        
        return prod_info
        
    except Exception as e:
        print(f"⚠️  Erreur récupération Production : {e}")
        return None


def get_models_by_year(client, target_year):
    """Récupère tous les modèles d'une année spécifique"""
    
    print("=" * 130)
    print(f"📊 ANALYSE DES MODÈLES - ANNÉE {target_year}")
    print("=" * 130)

    experiment_names = [
        'crime-prediction-baseline',
        'crime-prediction-ensemble'
    ]

    all_results = []

    for exp_name in experiment_names:
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            continue

        print(f"\n🔍 Experiment : {exp_name}")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            filter_string=f"params.year = '{target_year}'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=100
        )

        print(f"   → {len(runs)} runs trouvés")

        for run in runs:
            run_name = run.data.tags.get('mlflow.runName', 'N/A')
            model_type = run.data.params.get('model_type') or run.data.params.get('ensemble_type', 'N/A')

            test_acc = run.data.metrics.get('test_accuracy', 0)
            test_f1 = run.data.metrics.get('test_f1_weighted', 0)
            cv_mean = run.data.metrics.get('cv_accuracy_mean', 0)
            train_acc = run.data.metrics.get('train_accuracy', 0)

            all_results.append({
                'Type': 'Ensemble' if 'ensemble' in exp_name else 'Baseline',
                'Run Name': run_name,
                'Model': model_type,
                'Year': target_year,
                'Test Accuracy': test_acc,
                'Test F1': test_f1,
                'CV Mean': cv_mean,
                'Train Acc': train_acc,
                'Overfitting Gap': train_acc - test_acc if train_acc > 0 else 0,
                'Run ID': run.info.run_id
            })

    if not all_results:
        print(f"\n❌ Aucun modèle trouvé pour {target_year}")
        return None

    df = pd.DataFrame(all_results)
    df = df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    return df


def display_comparison_with_production(best_new_model, prod_info):
    """Affiche la comparaison avec Production"""
    print("\n" + "=" * 130)
    print("⚖️  COMPARAISON : NOUVEAU vs PRODUCTION (Information uniquement)")
    print("=" * 130)
    
    if prod_info is None:
        print("\n✅ Aucun modèle en Production → Promotion automatique")
        return
    
    new_acc = best_new_model['Test Accuracy']
    prod_acc = prod_info['test_accuracy']
    
    improvement = new_acc - prod_acc
    improvement_pct = (improvement / prod_acc) * 100 if prod_acc > 0 else 0
    
    print(f"\n📊 Modèle en Production Actuel (SERA REMPLACÉ) :")
    print(f"   • Version       : v{prod_info['version']}")
    print(f"   • Modèle        : {prod_info['model_type']}")
    print(f"   • Année         : {prod_info['year']}")
    print(f"   • Test Accuracy : {prod_acc:.4f} ({prod_acc*100:.2f}%)")
    print(f"   • Test F1       : {prod_info['test_f1']:.4f}")
    
    print(f"\n🆕 Meilleur Nouveau Modèle ({best_new_model['Year']}) (SERA PROMU) :")
    print(f"   • Modèle        : {best_new_model['Model']}")
    print(f"   • Test Accuracy : {new_acc:.4f} ({new_acc*100:.2f}%)")
    print(f"   • Test F1       : {best_new_model['Test F1']:.4f}")
    
    print(f"\n📈 Différence :")
    print(f"   • Δ Accuracy    : {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    if improvement > 0:
        print(f"\n✅ Le nouveau modèle est MEILLEUR (+{improvement_pct:.2f}%)")
    elif improvement == 0:
        print(f"\n➡️  Performance IDENTIQUE")
    else:
        print(f"\n⚠️  Le nouveau modèle est MOINS BON ({improvement_pct:.2f}%)")
    
    print(f"\n🔄 STRATÉGIE : REMPLACEMENT SYSTÉMATIQUE")
    print(f"   Le nouveau modèle sera promu QUEL QUE SOIT sa performance")
    print(f"   Raison : Utiliser toujours les données les plus récentes")


def display_comparison(df):
    """Affiche le classement des modèles de l'année"""
    print("\n" + "=" * 130)
    print(f"🏆 CLASSEMENT DES MODÈLES - ANNÉE {df.iloc[0]['Year']}")
    print("=" * 130)
    
    print(f"\n{'Rank':<5} {'Model':<20} {'Type':<10} {'Test Acc':<12} {'Test F1':<10} {'CV Mean':<10}")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        rank = idx + 1
        symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        
        print(f"{symbol} {rank:<3} {row['Model']:<20} {row['Type']:<10} "
              f"{row['Test Accuracy']:.4f}      {row['Test F1']:.4f}     {row['CV Mean']:.4f}")
    
    print("-" * 80)
    
    best = df.iloc[0]
    print(f"\n🏆 LE MEILLEUR DE {best['Year']} : {best['Model']} ({best['Test Accuracy']*100:.2f}%)")
    print(f"   → Ce modèle REMPLACERA celui en Production")


def create_dummy_input_from_features(features_str):
    """
    Crée un dummy input avec TOUTES les features utilisées lors de l'entraînement
    
    Args:
        features_str: String des features séparées par virgule (depuis MLflow params)
    
    Returns:
        DataFrame avec toutes les colonnes nécessaires
    """
    # Parser les features
    if not features_str:
        # Features par défaut si non trouvées
        features = ['Hour', 'Day_of_week', 'Month_num', 'LAT', 'LON', 'Vict Age', 'AREA']
        print(f"   ⚠️  Paramètre 'features' vide, utilisation des features par défaut")
    else:
        features = [f.strip() for f in features_str.split(',')]
    
    print(f"   📋 Features détectées ({len(features)}) : {features}")
    
    # Créer le DataFrame avec TOUTES les features
    dummy_data = {}
    
    for feat in features:
        # Valeurs par défaut pour chaque type de feature
        if feat == 'Hour':
            dummy_data[feat] = [12]
        elif feat == 'Day_of_week':
            dummy_data[feat] = [3]
        elif feat == 'Month_num':
            dummy_data[feat] = [6]
        elif feat == 'LAT':
            dummy_data[feat] = [34.05]
        elif feat == 'LON':
            dummy_data[feat] = [-118.25]
        elif feat == 'Vict Age':
            dummy_data[feat] = [35.0]
        elif feat == 'AREA':
            dummy_data[feat] = [15]
        elif feat == 'Vict Sex':
            dummy_data[feat] = [0]  # Valeur encodée
        elif feat == 'Vict Descent':
            dummy_data[feat] = [0]  # Valeur encodée
        elif feat == 'Premis Cd':
            dummy_data[feat] = [101.0]
        elif feat == 'Part 1-2':
            dummy_data[feat] = [1]
        else:
            # Feature inconnue, mettre une valeur par défaut
            dummy_data[feat] = [0]
            print(f"   ⚠️  Feature inconnue '{feat}', valeur par défaut = 0")
    
    df = pd.DataFrame(dummy_data)
    
    # S'assurer que l'ordre des colonnes correspond exactement
    df = df[features]
    
    print(f"   ✅ Dummy input créé : shape={df.shape}, colonnes={list(df.columns)}")
    
    return df


def promote_model(client, best_run_info, model_name="crime-prediction-model"):
    """
    Promouvoir le modèle en Production
    VERSION CORRIGÉE : Récupération dynamique des features
    """
    run_id = best_run_info['Run ID']
    model_type = best_run_info['Model']
    accuracy = best_run_info['Test Accuracy']
    year = best_run_info['Year']
    
    print("\n" + "=" * 130)
    print("🚀 PROMOTION FORCÉE EN PRODUCTION")
    print("=" * 130)
    
    print(f"\n🎯 Modèle à promouvoir :")
    print(f"   • Modèle        : {model_type}")
    print(f"   • Année         : {year}")
    print(f"   • Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Run ID        : {run_id[:12]}...")
    
    try:
        # 0. ÉTAPE CRITIQUE : Récupérer les métadonnées du run AVANT tout
        print(f"\n📋 Étape 0/5 : Récupération des métadonnées du run...")
        original_run = client.get_run(run_id)
        features_param = original_run.data.params.get('features', '')
        
        if not features_param:
            print("   ⚠️  Paramètre 'features' non trouvé dans MLflow")
        else:
            print(f"   ✅ Features récupérées depuis MLflow")
        
        # 1. Trouver le modèle
        print(f"\n📥 Étape 1/5 : Recherche du modèle...")
        artifacts = client.list_artifacts(run_id)
        joblib_files = [art.path for art in artifacts if art.path.endswith('.joblib')]
        
        if not joblib_files:
            print("❌ Aucun fichier .joblib trouvé")
            return False
        
        # Prioriser
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
        
        print(f"   ✅ Trouvé : {joblib_path}")
        
        # 2. Télécharger et charger
        print(f"\n📥 Étape 2/5 : Chargement...")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, joblib_path, dst_path=tmpdir)
            full_path = os.path.join(tmpdir, joblib_path)
            
            if full_path.endswith('_artifacts.joblib'):
                artifacts_bundle = joblib.load(full_path)
                model = artifacts_bundle.get('model', artifacts_bundle)
            else:
                model = joblib.load(full_path)
            
            print(f"   ✅ Chargé : {type(model).__name__}")
            
            # 3. Créer signature avec les BONNES features
            print(f"\n🔧 Étape 3/5 : Création de la signature...")
            
            # ⭐ CORRECTION CRITIQUE : Créer dummy input avec TOUTES les features
            dummy_input = create_dummy_input_from_features(features_param)
            
            # Tester la prédiction AVANT de créer la signature
            print(f"   🧪 Test de prédiction...")
            try:
                predictions = model.predict(dummy_input)
                print(f"   ✅ Prédiction réussie : {predictions}")
            except Exception as pred_error:
                print(f"   ❌ ERREUR de prédiction : {pred_error}")
                print(f"   📊 Détails du problème :")
                print(f"      - Shape dummy_input : {dummy_input.shape}")
                print(f"      - Colonnes : {list(dummy_input.columns)}")
                print(f"      - Dtypes : {dummy_input.dtypes.to_dict()}")
                print(f"      - Type modèle : {type(model).__name__}")
                
                # Si c'est un modèle LightGBM, afficher plus d'infos
                if 'LightGBM' in str(type(model).__name__) or 'LGBM' in str(type(model).__name__):
                    try:
                        print(f"      - Nombre de features attendues par le modèle : {model.n_features_}")
                    except:
                        pass
                
                raise pred_error
            
            # Créer la signature
            signature = infer_signature(dummy_input, predictions)
            print(f"   ✅ Signature créée avec succès")
            
            # 4. Enregistrer dans MLflow
            print(f"\n📝 Étape 4/5 : Enregistrement dans MLflow...")
            
            with mlflow.start_run(run_name=f"promote_force_{year}_{run_id[:8]}"):
                # Copier les métriques
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
                mlflow.set_tag("year", year)
                mlflow.set_tag("promotion_strategy", "always_replace")
                
                # Enregistrer le modèle
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=dummy_input,
                    registered_model_name=model_name
                )
            
            print(f"   ✅ Enregistré dans le Model Registry")
        
        # 5. Récupérer nouvelle version
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            print("❌ Version non créée")
            return False
        
        new_version = latest_versions[0].version
        print(f"   → Nouvelle version créée : v{new_version}")
        
        # 6. Archiver anciennes versions Production
        print(f"\n📦 Étape 5/5 : Transition vers Production...")
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        for v in prod_versions:
            try:
                old_run = client.get_run(v.run_id)
                old_year = old_run.data.params.get('year', 'Unknown')
                old_acc = old_run.data.metrics.get('test_accuracy', 0)
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived"
                )
                print(f"   ✓ v{v.version} (année {old_year}, acc={old_acc:.4f}) → Archived")
            except Exception as e:
                print(f"   ⚠️  Erreur lors de l'archivage de v{v.version} : {e}")
        
        # 7. Promouvoir en Production
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production"
        )
        
        print(f"   ✓ v{new_version} (année {year}, acc={accuracy:.4f}) → PRODUCTION")
        
        # 8. Ajouter une description détaillée
        description = f"""
🔄 REMPLACEMENT SYSTÉMATIQUE - Année {year}

📊 Métriques de Performance :
   • Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)
   • Test F1-Score : {best_run_info['Test F1']:.4f}
   • CV Mean       : {best_run_info['CV Mean']:.4f}

🔧 Configuration :
   • Type de modèle : {best_run_info['Type']}
   • Algorithme     : {model_type}
   • Année données  : {year}
   • Nombre features: {len(features_param.split(',')) if features_param else 'N/A'}

📅 Informations de Promotion :
   • Date promotion : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
   • Run ID original: {run_id}
   • Stratégie      : Remplacement systématique (toujours utiliser données les plus récentes)
   
⚡ Note : Ce modèle remplace l'ancien INDÉPENDAMMENT de sa performance relative.
          La stratégie privilégie l'utilisation des données les plus récentes.
        """.strip()
        
        client.update_model_version(
            name=model_name,
            version=new_version,
            description=description
        )
        
        print("\n" + "=" * 130)
        print("✅ PROMOTION RÉUSSIE !")
        print("=" * 130)
        print(f"\n🎉 {model_name} v{new_version} est maintenant en PRODUCTION")
        print(f"   • Année    : {year}")
        print(f"   • Accuracy : {accuracy*100:.2f}%")
        print(f"   • Modèle   : {model_type}")
        print(f"\n📍 Vérifier sur DagsHub : https://dagshub.com/benrhoumamohamed752/ProjetMLOps")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE : {e}")
        import traceback
        print("\n📋 Traceback complet :")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Promotion FORCÉE du meilleur modèle de l\'année')
    parser.add_argument('--year', type=str, required=True, help='Année à analyser')
    parser.add_argument('--save', action='store_true', help='Sauvegarder rapport')
    parser.add_argument('--auto_promote', action='store_true', help='Promotion automatique')
    parser.add_argument('--model_name', type=str, default='crime-prediction-model')
    args = parser.parse_args()

    client = connect_to_mlflow()

    # 1. Récupérer le modèle en Production actuel (pour info)
    print("\n" + "=" * 130)
    print("🔍 MODÈLE EN PRODUCTION ACTUEL (SERA REMPLACÉ)")
    print("=" * 130)
    
    prod_info = get_production_model_info(client, args.model_name)
    
    if prod_info:
        print(f"\n📊 Actuellement en Production :")
        print(f"   • Version : v{prod_info['version']}")
        print(f"   • Modèle  : {prod_info['model_type']}")
        print(f"   • Année   : {prod_info['year']}")
        print(f"   • Accuracy: {prod_info['test_accuracy']:.4f} ({prod_info['test_accuracy']*100:.2f}%)")
    else:
        print("\n✅ Aucun modèle en Production → Premier déploiement")
    
    # 2. Récupérer les modèles de la nouvelle année
    df = get_models_by_year(client, args.year)
    
    if df is None or len(df) == 0:
        print(f"\n❌ Aucun modèle trouvé pour {args.year}")
        print(f"   Vérifiez que les modèles ont bien été entraînés avec le paramètre year='{args.year}'")
        return
    
    # 3. Afficher le classement
    display_comparison(df)
    
    # 4. Meilleur modèle de la nouvelle année
    best_new = df.iloc[0].to_dict()
    
    # 5. Afficher la comparaison (information uniquement)
    display_comparison_with_production(best_new, prod_info)
    
    # 6. Sauvegarder si demandé
    if args.save:
        os.makedirs('reports', exist_ok=True)
        report_path = f'reports/models_comparison_{args.year}.csv'
        df.to_csv(report_path, index=False)
        print(f"\n💾 Rapport sauvegardé : {report_path}")
    
    # 7. TOUJOURS promouvoir si auto_promote
    if args.auto_promote:
        print(f"\n" + "=" * 130)
        print(f"🔄 STRATÉGIE : REMPLACEMENT SYSTÉMATIQUE")
        print(f"=" * 130)
        print(f"\nLe meilleur modèle de {args.year} REMPLACERA celui en Production")
        print(f"QUEL QUE SOIT sa performance relative")
        
        success = promote_model(client, best_new, args.model_name)
        
        if success:
            print(f"\n🎊 SUCCÈS : Modèle {args.year} déployé en Production !")
        else:
            print(f"\n❌ ÉCHEC : La promotion a échoué - vérifier les logs ci-dessus")
            exit(1)
    else:
        print(f"\n💡 Pour promouvoir automatiquement :")
        print(f"   python src/models/promote_best_model.py --year {args.year} --auto_promote")
    
    print("\n" + "=" * 130)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 130)


if __name__ == "__main__":
    main()