import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the saved model
model = load_model("models/churn_model")

# Load cleaned training dataset to use as template
template_df = pd.read_csv("data/clean/cleaned_telco.csv")

def predict_churn_with_feedback(gender, senior_citizen, partner, tenure, total_charges, contract, internet_service, feedback):
    input_df = template_df.sample(1).copy()

    input_df["gender_Female"] = 1 if gender == "Female" else 0
    input_df["gender_Male"] = 1 if gender == "Male" else 0
    input_df["SeniorCitizen"] = int(senior_citizen)
    input_df["Partner"] = int(partner)
    input_df["tenure"] = int(tenure)
    input_df["TotalCharges"] = float(total_charges)

    for val in ["DSL", "Fiber optic", "No"]:
        input_df[f"InternetService_{val}"] = 1 if internet_service == val else 0

    for val in ["Month-to-month", "One year", "Two year"]:
        input_df[f"Contract_{val}"] = 1 if contract == val else 0

    result = predict_model(model, data=input_df)
    print("📊 Prediction result columns:", result.columns.tolist())  # Debug

    try:
        label = result["prediction_label"].iloc[0]
        score = result["prediction_score"].iloc[0]
    except KeyError:
        print("❌ Column names not found! Actual columns:", result.columns)
        return "Prediction failed due to column mismatch."

    return f"🧠 Prediction: {'Yes' if label == 1 else 'No'} (Confidence: {score:.2%})"


# Build the Gradio UI
demo = gr.Interface(
    fn=predict_churn_with_feedback,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["0", "1"], label="Senior Citizen"),
        gr.Dropdown(["1", "0"], label="Partner (1=Yes, 0=No)"),
        gr.Slider(0, 72, step=1, label="Tenure (months)"),
        gr.Number(label="Total Charges"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Radio(["Correct", "Incorrect"], label="Was this prediction correct?")
    ],
    outputs="text",
    title="🧠 Smart Churn Predictor",
    description="This model learns from your feedback to get better over time."
)

if __name__ == "__main__":
    demo.launch()
