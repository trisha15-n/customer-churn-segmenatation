import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

pipeline = PredictPipeline()

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("Telco Customer Churn Prediction")
st.markdown("Fill the customer details to predict whether they are likely to churn or not.")

col1, col2 = st.columns(2)

with col1:
  gender = st.selectbox("Gender", ["Male", "Female"])
  SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
  Partner = st.selectbox("Partner", ["Yes", "No"])
  Dependents = st.selectbox("Dependents",["Yes", "No"])
  tenure = st.number_input("Tenure(months)", min_value=0, max_value=100, value=12)
  PhoneService = st.selectbox("Phone Service", ['Yes','No'])
  MultipleLines = st.selectbox("Multiple Lines",["Yes","No", 'No Phone Service'])
  InternetService = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])

with col2:
  OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No Internet Service"])
  OnlineBackup = st.selectbox("Online Backup", ["Yes","No", "No Internet Service"])
  DeviceProtection = st.selectbox("Device Protection", ['Yes', "No", "No Internet Service"])
  TechSupport = st.selectbox("Tech Support", ['Yes', "No", "No Internet Service"])
  StreamingTV = st.selectbox("Streaming TV", ['Yes', "No", "No Internet Service"])
  StreamingMovies = st.selectbox("Streaming Movies", ['Yes', "No", "No Internet Service"])
  Contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
  PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", 'No'])
  PaymentMethod = st.selectbox("Payment Menthod", ['Electronic check', "Mailed check", "Bank Transfer(automatic)","Credit card(automatic)"])


st.subheader("Charges") 
MonthlyCharges = st.number_input("Monthly Charges in $", min_value=0.0, max_value=1000.0, value=70.0)
TotalCharges = st.number_input("Total Charges in $", min_value=0.0, max_value=100000.0, value=500.0)

input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

if st.button("Predict"):
  try:
    prediction = pipeline.predict(input_data)
    if prediction == "Yes":
      st.error("This customer is likely to churn.")
    else:
      st.error("This customer is unlikely to churn.")
  except Exception as e:
    st.exception(f"Error: {e}")
