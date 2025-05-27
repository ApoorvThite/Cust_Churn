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

