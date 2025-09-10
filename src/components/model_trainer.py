import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

class ModelTrainer:
  def __init__(self):
    self.model_file_path = os.path.join("artifacts","model.pkl")

  def evaluate_model(self, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="Yes") 
    roc = roc_auc_score((y_test == 'Yes').astype(int), (y_pred == "Yes").astype(int))

    return {
      "model":model,
      "accuracy":accuracy,
      "f1":f1,
      "roc_auc":roc,
      "report":classification_report(y_test, y_pred)
    } 
  
  def initiate_model_trainer(self, train_path, test_path):
    try:
      transformer = DataTransformation()
      X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)

      logging.info("Starting Model Training.")

      models = {
        "LogisticRegression":LogisticRegression(max_iter=1000, solver='liblinear'),
        "RandomForest":RandomForestClassifier(n_estimators=200, random_state=42)
      }

      results = {}
      best_model_name = None
      best_score = -1
      best_model = None

      for name, model in models.items():
        logging.info(f"Training {name}...")
        metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = metrics

        logging.info(f"{name} -> Acc: {metrics['accuracy']:.3f}, "f"F1: {metrics['f1']:.3f}, ROC AUC: {metrics['roc_auc']:.3f}")

        if metrics["f1"] > best_score:
          best_score = metrics['f1']
          best_model_name = name
          best_model = metrics["model"]
      os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
      with open(self.model_file_path, "wb") as f:
        pickle.dump(best_model, f)

      logging.info(f"Model saved at {self.model_file_path}")

      return results, best_model_name, self.model_file_path
    except Exception as e:
      raise CustomException(e, sys)
    
if __name__ == "__main__":
  project_root = os.path.abspath(".")
  train_path = os.path.join(project_root, "artifacts", "train.csv")
  test_path = os.path.join(project_root, "artifacts", "test.csv")

  trainer = ModelTrainer()
  results, best_model_name, model_path = trainer.initiate_model_trainer(train_path, test_path)

  print("Model Training Summary :")
  for name, metrics in results.items():
    print(f"\n{name}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(metrics["report"])

  print(f"\n Best Model : {best_model_name} saved at {model_path}")  
      