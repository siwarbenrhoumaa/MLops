"""
Script de comparaison complète de TOUS les modèles
6 modèles : 4 baseline (RF, XGB, LGBM, LogReg) + 2 ensemble (Voting, Stacking)
"""

import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import argparse


def connect_to_mlflow():
    """Connecte à MLflow via DagsHub"""
    dagshub.init(repo_owner='benrhoumamohamed752', repo_name='ProjetMLOps', mlflow=True)
    client = MlflowClient()
    print("✅ Connecté à MLflow via DagsHub\n")
    return client


def get_all_models_comparison(client):
    """
    Récupère TOUS les modèles des 2 experiments et les compare
    """
    print("=" * 130)
    print("📊 COMPARAISON COMPLÈTE DE TOUS LES MODÈLES (2020)")
    print("=" * 130)
    
    # Les 2 experiments à analyser
    experiments = {
        'Baseline': 'crime-prediction-2020',
        'Ensemble': 'crime-prediction-ensemble-2020'
    }
    
    all_results = []
    
    for exp_type, exp_name in experiments.items():
        print(f"\n🔍 Analyse de l'experiment : {exp_name}")
        
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            print(f"   ⚠️ Experiment '{exp_name}' non trouvé, skip...")
            continue
        
        # Récupérer TOUS les runs (pas de limite)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["metrics.test_accuracy DESC"],
            max_results=100  # Large pour tout récupérer
        )
        
        print(f"   → {len(runs)} runs trouvés")
        
        for run in runs:
            run_name = run.data.tags.get('mlflow.runName', 'N/A')
            model_type = run.data.params.get('model_type', 
                                            run.data.params.get('ensemble_type', 'N/A'))
            
            # Métriques
            test_acc = run.data.metrics.get('test_accuracy', 0)
            test_f1 = run.data.metrics.get('test_f1_weighted', 
                                          run.data.metrics.get('test_f1', 0))
            cv_mean = run.data.metrics.get('cv_accuracy_mean', 0)
            cv_std = run.data.metrics.get('cv_accuracy_std', 0)
            train_acc = run.data.metrics.get('train_accuracy', 0)
            
            # Calculer l'overfitting gap
            overfitting_gap = train_acc - test_acc if train_acc > 0 else 0
            
            all_results.append({
                'Type': exp_type,
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
    
    # Créer le DataFrame
    if not all_results:
        print("\n❌ Aucun modèle trouvé !")
        return None
    
    df = pd.DataFrame(all_results)
    
    # Trier par Test Accuracy (décroissant)
    df = df.sort_values('Test Accuracy', ascending=False)
    
    return df


def display_comparison(df):
    """
    Affiche une comparaison complète et détaillée
    """
    print("\n" + "=" * 130)
    print("🏆 CLASSEMENT COMPLET DE TOUS LES MODÈLES")
    print("=" * 130)
    
    # Affichage formaté
    print(f"\n{'Rank':<5} {'Type':<10} {'Model':<20} {'Run Name':<35} {'Test Acc':<12} {'Test F1':<10} {'CV Mean':<10} {'Overfit':<10}")
    print("-" * 130)
    
    for idx, row in df.iterrows():
        rank = df.index.get_loc(idx) + 1
        
        # Symbole pour le meilleur
        symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        
        # Couleur pour overfitting
        overfit_symbol = "⚠️" if row['Overfitting Gap'] > 0.05 else "✅"
        
        print(f"{symbol} {rank:<3} {row['Type']:<10} {row['Model']:<20} {row['Run Name'][:34]:<35} "
              f"{row['Test Accuracy']:<12.4f} {row['Test F1']:<10.4f} {row['CV Mean']:<10.4f} "
              f"{overfit_symbol} {row['Overfitting Gap']:<9.4f}")
    
    print("-" * 130)


def display_statistics(df):
    """
    Affiche des statistiques détaillées
    """
    print("\n" + "=" * 130)
    print("📈 STATISTIQUES DÉTAILLÉES")
    print("=" * 130)
    
    # Statistiques par type
    print("\n📊 Moyennes par Type :")
    print("-" * 60)
    stats_by_type = df.groupby('Type').agg({
        'Test Accuracy': ['mean', 'std', 'min', 'max'],
        'Test F1': 'mean',
        'Overfitting Gap': 'mean'
    }).round(4)
    print(stats_by_type)
    
    # Statistiques par modèle
    print("\n📊 Moyennes par Modèle :")
    print("-" * 60)
    stats_by_model = df.groupby('Model').agg({
        'Test Accuracy': ['mean', 'count'],
        'Test F1': 'mean',
        'CV Mean': 'mean'
    }).round(4)
    print(stats_by_model)


def display_best_model_details(client, df):
    """
    Affiche les détails complets du meilleur modèle
    """
    best = df.iloc[0]
    
    print("\n" + "=" * 130)
    print("🏆 MEILLEUR MODÈLE - DÉTAILS COMPLETS")
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
    print(f"   • CV Std           : {best['CV Std']:.4f}")
    print(f"   • Train Accuracy   : {best['Train Acc']:.4f}")
    
    print(f"\n⚖️ Analyse de Stabilité :")
    gap = best['Overfitting Gap']
    if gap > 0.1:
        status = "⚠️ OVERFITTING DÉTECTÉ"
        advice = "→ Considérer plus de régularisation ou plus de données"
    elif gap > 0.05:
        status = "⚠️ LÉGER OVERFITTING"
        advice = "→ Acceptable, mais à surveiller"
    else:
        status = "✅ BON ÉQUILIBRE"
        advice = "→ Modèle stable et généralisable"
    
    print(f"   • Overfitting Gap  : {gap:.4f}")
    print(f"   • Statut           : {status}")
    print(f"   • Recommandation   : {advice}")
    
    # Récupérer les paramètres du run
    try:
        run = client.get_run(best['Run ID'])
        
        print(f"\n🔧 Configuration du Modèle :")
        important_params = ['model_type', 'ensemble_type', 'n_classes', 'target', 
                           'features', 'voting', 'meta_learner']
        for param in important_params:
            value = run.data.params.get(param)
            if value:
                print(f"   • {param:15} : {value}")
    except:
        pass
    
    # Comparaison avec les autres
    print(f"\n📈 Comparaison avec les Autres :")
    if len(df) > 1:
        second_best = df.iloc[1]
        gap_with_second = best['Test Accuracy'] - second_best['Test Accuracy']
        print(f"   • 2ème meilleur    : {second_best['Model']} ({second_best['Test Accuracy']:.4f})")
        print(f"   • Écart            : +{gap_with_second:.4f} ({gap_with_second*100:.2f}%)")
        
        if gap_with_second > 0.01:
            print(f"   • Verdict          : ✅ Nettement supérieur")
        elif gap_with_second > 0.005:
            print(f"   • Verdict          : ✅ Légèrement supérieur")
        else:
            print(f"   • Verdict          : ⚖️ Performance similaire")


def display_top_3(df):
    """
    Affiche le podium des 3 meilleurs
    """
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


def save_comparison_report(df, filename='reports/models_comparison_2020.csv'):
    """
    Sauvegarde le rapport de comparaison
    """
    import os
    os.makedirs('reports', exist_ok=True)
    
    df.to_csv(filename, index=False)
    print(f"\n💾 Rapport sauvegardé : {filename}")


def recommend_action(df):
    """
    Recommande l'action à prendre
    """
    best = df.iloc[0]
    
    print("\n" + "=" * 130)
    print("💡 RECOMMANDATION FINALE")
    print("=" * 130)
    
    print(f"\n🎯 Modèle Recommandé : {best['Model'].upper()}")
    print(f"   Run Name : {best['Run Name']}")
    print(f"   Accuracy : {best['Test Accuracy']:.4f} ({best['Test Accuracy']*100:.2f}%)")
    
    print(f"\n🚀 Prochaines Étapes :")
    print(f"   1️⃣  Promouvoir ce modèle en production :")
    print(f"       python promote_best_model_2020.py \\")
    print(f"           --experiment {df.iloc[0]['Type'].lower()}-2020 \\")
    print(f"           --auto_promote")
    
    print(f"\n   2️⃣  Vérifier sur DagsHub :")
    print(f"       https://dagshub.com/benrhoumamohamed752/ProjetMLOps")
    
    print(f"\n   3️⃣  Tester le modèle :")
    print(f"       python use_production_model.py --mode demo")
    
    print(f"\n   4️⃣  Déployer :")
    print(f"       uvicorn src.deployment.api:app --reload")
    print(f"       streamlit run src/deployment/streamlit_app.py")
    
    # Avertissement si overfitting
    if best['Overfitting Gap'] > 0.05:
        print(f"\n⚠️  ATTENTION : Overfitting détecté (gap = {best['Overfitting Gap']:.4f})")
        print(f"   → Considérer plus de régularisation")
        print(f"   → Ou ajouter plus de données (combiner avec 2021)")


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
    parser = argparse.ArgumentParser(description='Comparer TOUS les modèles')
    parser.add_argument('--save', action='store_true', help='Sauvegarder le rapport en CSV')
    parser.add_argument('--top', type=int, default=None, help='Afficher seulement le top N')
    parser.add_argument('--promote', action='store_true', help='Promouvoir automatiquement le meilleur modèle')
    parser.add_argument('--auto_promote', action='store_true', help='Promouvoir sans confirmation')
    parser.add_argument('--model_name', type=str, default='crime-prediction-model',
                       help='Nom du modèle dans le Registry')
    args = parser.parse_args()
    
    # Connexion
    client = connect_to_mlflow()
    
    # Récupérer tous les modèles
    df = get_all_models_comparison(client)
    
    if df is None:
        return
    
    # Filtrer si demandé
    if args.top:
        df = df.head(args.top)
        print(f"\n📌 Affichage limité au Top {args.top}")
    
    # Affichages
    display_comparison(df)
    display_top_3(df)
    display_best_model_details(client, df)
    display_statistics(df)
    
    # Sauvegarder si demandé
    if args.save:
        save_comparison_report(df)
    
    # Recommandation
    recommend_action(df)
    
    # Promotion si demandé
    if args.promote or args.auto_promote:
        best_model_info = df.iloc[0].to_dict()
        success = promote_best_model(
            client, 
            best_model_info, 
            model_name=args.model_name,
            auto=args.auto_promote
        )
        
        if success:
            print("\n🚀 Prochaines étapes :")
            print("   1. Tester : python use_production_model.py --mode demo")
            print("   2. API    : uvicorn src.deployment.api:app --reload")
            print("   3. UI     : streamlit run src/deployment/streamlit_app.py")
    
    print("\n" + "=" * 130)
    print("✅ COMPARAISON TERMINÉE")
    print("=" * 130)


if __name__ == "__main__":
    main()
