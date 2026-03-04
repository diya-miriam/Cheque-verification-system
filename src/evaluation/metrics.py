import pandas as pd
import mlflow

def log_confusion_matrix(cm, file_name="confusion_matrix.csv", artifact_path="evaluation"):
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Negative", "Actual_Positive"],
        columns=["Predicted_Negative", "Predicted_Positive"],
    )
    cm_df.to_csv(file_name)
    mlflow.log_artifact(file_name, artifact_path=artifact_path)