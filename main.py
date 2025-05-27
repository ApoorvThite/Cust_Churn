import polars as pl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import clean_telco_data, DataIntegrityReporter
from src.train_model import train_and_save_model


# Load raw dataset
df = pl.read_csv("data/raw/telco_churn.csv")

# Run data integrity report before cleaning
print("🔍 Running Data Integrity Check...")
reporter = DataIntegrityReporter(df)
reporter.summary_report()
reporter.class_imbalance_report()

# Clean and transform
cleaned_df = clean_telco_data(df)

# Optional: Save cleaned data
os.makedirs("data/clean", exist_ok=True)
cleaned_df.write_csv("data/clean/cleaned_telco.csv")

print("\n✅ Data cleaning complete. Cleaned dataset shape:", cleaned_df.shape)


# Assuming cleaned CSV was saved in Step 2
cleaned_data_path = "data/clean/cleaned_telco.csv"

# Train and save model
train_and_save_model(cleaned_data_path)
