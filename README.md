# 🧠 Customer Churn Predictor

A machine learning-powered web application that predicts whether a customer is likely to churn, based on their profile and service usage. Built with PyCaret, Gradio, and Polars, this tool helps businesses make informed decisions to retain customers more effectively.

---

## 🚀 Project Overview

This project takes a real-world Telco customer dataset, cleans and analyzes it, builds a classification model to predict churn, and deploys the model through an interactive web interface. Users can input customer attributes, get churn predictions, and provide feedback to help improve the system.

---

## 📦 Tech Stack

| Category        | Tools Used                                              |
|----------------|----------------------------------------------------------|
| Data Handling   | `Polars`, `Pandas`                                      |
| Model Training  | `PyCaret (Classification Module)`                        |
| Model Deployment| `Gradio`                                                |
| Environment     | Python 3.10 (via `venv`)                                |

---

## 📁 Project Structure

```bash
ChurnPredictor/
├── app/
│ └── gradio_ui.py # Gradio web UI for interacting with the model
├── churn-pycaret-310/ # Virtual environment (excluded from Git)
├── data/
│ ├── raw/
│ │ └── telco_churn.csv # Original Telco dataset
│ └── clean/
│ └── cleaned_telco.csv # Cleaned dataset
├── models/
│ └── churn_model.pkl # Saved trained model
├── notebooks/ # (Optional) For exploratory notebooks
├── src/
│ ├── preprocessing.py # Data integrity and cleaning logic
│ ├── train_model.py # Model training logic using PyCaret
│ ├── predict.py # Custom prediction utilities (if any)
├── main.py # Pipeline runner: load, clean, train
├── requirements.txt # Dependencies
├── .gitignore # Ignored files and directories
└── README.md # You are here!
```

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ApoorvThite/Cust_Churn.git
cd Cust_Churn

# Create and Activate the Virtual Environment
python3.10 -m venv churn-pycaret-310
source churn-pycaret-310/bin/activate

# Install Dependencies
pip install -r requirements.txt
```
----------------------------------------------------

Step 1: Clean Data & Train Model
```
python main.py
```
This will:

Clean the Telco dataset

Save the cleaned CSV

Train and compare models using PyCaret

Save the best model in models/churn_model.pkl
--------------------------------------------------------

Step 2: Launch the Gradio App
```
python app/gradio_ui.py
```

Visit: http://127.0.0.1:7860 (or similar) to use the app.

--------------------------------------------------------

🧠 How It Works
* Users input customer attributes (gender, tenure, contract type, etc.).

* The model predicts the likelihood of churn.

* Users give feedback if the prediction was accurate or not (future improvement area).

* The pipeline is built with modularity in mind, allowing future enhancements such as SHAP explainability or GPT-based explanation.
