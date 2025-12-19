import argparse
import pandas as pd
import os
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

def detect_drift(old_file, new_file, threshold=0.1):
    print(f"Chargement des données : {old_file} et {new_file}")
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)

    # Colonnes communes
    common_columns = old_df.columns.intersection(new_df.columns)
    old_df = old_df[common_columns]
    new_df = new_df[common_columns]
    print(f"Colonnes communes utilisées : {list(common_columns)}")

    # Métriques
    metrics = [DatasetDriftMetric()]

    # Colonnes numériques sûres pour drift individuel
    safe_numeric_cols = ['LAT', 'LON', 'Vict Age', 'AREA', 'Premis Cd', 'Hour', 'Day_of_week', 'Month_num']
    for col in safe_numeric_cols:
        if col in common_columns:
            metrics.append(ColumnDriftMetric(column_name=col))

    # Drift sur la cible si elle existe
    if 'Crime_Group' in common_columns:
        metrics.append(ColumnDriftMetric(column_name='Crime_Group'))
        print("Drift calculé sur Crime_Group")
    else:
        print("Attention : 'Crime_Group' absente → drift cible ignoré")

    # Génération du rapport
    report = Report(metrics=metrics)
    report.run(reference_data=old_df, current_data=new_df)

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    report.save_html(report_path)
    print(f"Rapport généré : {os.path.abspath(report_path)}")

    # Extraction robuste du drift score
    result = report.as_dict()
    dataset_metric = result['metrics'][0]['result']

    # Clés possibles selon la version/config
    if 'drift_score' in dataset_metric:
        drift_score = dataset_metric['drift_score']
    elif 'drift_by_columns' in dataset_metric:
        # Cas alternatif
        drift_score = dataset_metric.get('dataset_drift', False)  # fallback
    else:
        drift_score = 0.5  # fallback sécurisé

    drift_detected = drift_score > threshold

    print(f"Drift score global : {drift_score:.3f}")
    print(f"Drift détecté (threshold={threshold}) : {drift_detected}")

    # Pour GitHub Actions / test_pipeline
    with open(os.environ.get("GITHUB_OUTPUT", "drift_output.txt"), "w") as f:
        f.write(f"drift_detected={str(drift_detected).lower()}\n")

    return drift_detected

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_year', type=int, default=2020)
    parser.add_argument('--new_year', type=int, required=True)
    args = parser.parse_args()

    old_file = f"data/processed/crime_{args.old_year}_processed.csv"
    new_file = f"data/processed/crime_{args.new_year}_processed.csv"

    if not os.path.exists(old_file):
        raise FileNotFoundError(f"Fichier manquant : {old_file}")
    if not os.path.exists(new_file):
        raise FileNotFoundError(f"Fichier manquant : {new_file}")

    detect_drift(old_file, new_file, threshold=0.1)