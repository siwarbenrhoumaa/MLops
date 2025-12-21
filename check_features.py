import os
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/benrhoumamohamed752/ProjetMLOps.mlflow"

client = MlflowClient()
exp = client.get_experiment_by_name("crime-prediction-baseline")

if exp:
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="params.year = '2022'",
        max_results=1
    )
    if runs:
        run = runs[0]
        print("Run trouvé :", run.info.run_id)
        print("Features :", run.data.params.get("features", "NON TROUVÉ"))
    else:
        print("Aucun run trouvé pour 2022")
else:
    print("Experiment non trouvé")
