import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model
import os

def train_and_save_model(data_path: str, target_col: str = "Churn", output_dir: str = "models/"):
    print("🚀 Loading cleaned dataset...")
    df = pd.read_csv(data_path)

    print("📊 Initializing PyCaret setup...")

    setup(
        data=df,
        target=target_col,
        session_id=42,
        use_gpu=True
    )

    print("⚙️ Training and comparing models...")
    best_model = compare_models()

    print("✅ Finalizing and saving best model...")
    final_model = finalize_model(best_model)

    os.makedirs(output_dir, exist_ok=True)
    save_model(final_model, os.path.join(output_dir, "churn_model"))

    print(f"📁 Model saved to: {output_dir}/churn_model.pkl")
