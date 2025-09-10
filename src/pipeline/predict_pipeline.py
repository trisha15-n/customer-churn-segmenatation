import os
import pickle
import pandas as pd

class PredictPipeline:
  def __init__(self):
    self.model_path = os.path.join("artifacts", "model.pkl")
    self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    with open(self.preprocessor_path, 'rb') as f:
      self.preprocessor = pickle.load(f)
    with open(self.model_path, 'rb') as f:
      self.model = pickle.load(f)

  def predict(self, input_data: dict):
    df = pd.DataFrame([input_data])
    if "customerID" in df.columns:
      df = df.drop(columns=["customerID"])

    transformed = self.preprocessor.transform(df)
    prediction = self.model.predict(transformed)
    return prediction[0]
  

if __name__ == "__main__":
  sample_input = {
    
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 8,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.35,
    "TotalCharges": 605.45
    }
  pipeline = PredictPipeline()
  result = pipeline.predict(sample_input)
  print("Churn Prediction:", result) 

