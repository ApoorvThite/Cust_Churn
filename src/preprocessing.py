import polars as pl

def clean_telco_data(df: pl.DataFrame) -> pl.DataFrame:
    print("🧼 Running clean_telco_data...")

    # Drop unnecessary column
    df = df.drop("customerID")

    # Fix data types
    df = df.with_columns([
        pl.col("TotalCharges").cast(pl.Float64),
        pl.col("SeniorCitizen").cast(pl.Utf8),
    ])

    # Drop nulls
    df = df.drop_nulls()

    # Encode binary features
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    df = df.with_columns([
        pl.when(pl.col(col) == "Yes").then(1).otherwise(0).alias(col)
        for col in binary_cols
    ])

    # One-hot encode categorical columns
    cat_cols = [
        "gender", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaymentMethod"
    ]
    df = df.to_dummies(columns=cat_cols)

    print("✅ Cleaning complete.")
    return df

class DataIntegrityReporter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def summary_report(self):
        print("\n📊 Column Overview:")
        print(self.df.describe())

        print("\n🚨 Missing Values:")
        nulls = self.df.null_count()
        print(nulls.filter(pl.col(nulls.columns[0]) > 0))

        print("\n🧪 Unique Values:")
        for col in self.df.columns:
            unique_count = self.df.select(pl.col(col).n_unique()).item()
            if unique_count < 10:
                uniques = self.df.select(pl.col(col).unique()).to_series()
                print(f"{col} → {uniques}")

    def class_imbalance_report(self, target_col="Churn"):
        print(f"\n📉 Class Distribution for '{target_col}':")

    # Value counts returns a struct column — alias and unnest
        vc_df = self.df.select(
            pl.col(target_col).value_counts().alias("vc")
        ).unnest("vc")

    # Optional: rename only target_col for cleaner output
        vc_df = vc_df.rename({target_col: "value"})

    # Sort and print
        vc_sorted = vc_df.sort("count", descending=True)
        print(vc_sorted)

print("✅ preprocessing.py is loaded.")


