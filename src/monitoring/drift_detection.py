import argparse
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

def detect_drift(old_file, new_file, threshold=0.1):
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)

    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name='LAT'),  # Exemple de colonnes à checker
        ColumnDriftMetric(column_name='LON'),
        ColumnDriftMetric(column_name='Crime_Group')
    ])
    report.run(reference_data=old_df, current_data=new_df)
    report.save_html("reports/drift_report.html")

    drift_share = report.as_dict()['metrics'][0]['result']['drift_share']
    print(f"Drift share: {drift_share}")
    return drift_share > threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_year', type=int, default=2020)
    parser.add_argument('--new_year', type=int, required=True)
    args = parser.parse_args()

    old_file = f"data/processed/crime_{args.old_year}_processed.csv"
    new_file = f"data/processed/crime_{args.new_year}_processed.csv"

    drift = detect_drift(old_file, new_file)
    print(f"drift_detected={drift}")
    # Pour GitHub Actions
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"drift_detected={drift}\n")