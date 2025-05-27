import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model
import os

def retrain_from_feedback(feedback_file: str, output_path: str = "models/churn_model"):
    print("📥 Loading feedback data...")
    df = pd.read_csv(feedback_file)

    # Use only rows where the user gave feedback
    df = df[df["UserFeedback"].notnull()]
    
    # Create corrected labels
    df["CorrectedLabel"] = df.apply(lambda row: row["PredictedChurn"] if row["UserFeedback"] == "Correct" else 1 - row["PredictedChurn"], axis=1)

    # Drop prediction & feedback columns
    df_clean = df.drop(columns=["PredictedChurn", "Confidence", "UserFeedback"])

    print("🔄 Retraining with feedback-corrected labels...")
    clf = setup(df_clean, target="CorrectedLabel", session_id=101, silent=True, verbose=False)
    best = compare_models()
    final = finalize_model(best)

    save_model(final, output_path)
    print(f"✅ Retrained model saved at: {output_path}.pkl")