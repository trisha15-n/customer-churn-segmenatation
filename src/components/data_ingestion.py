import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

class DataIngestionConfig:
  raw_data_path: str = os.path.join("artifacts", "data.csv")
  train_data_path: str = os.path.join("artifacts","train.csv")
  test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
  def __init__(self):
    self.ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self):
    logging.info("Starting data ingestion.")
    try:
      df = pd.read_csv(os.path.join("data","data.csv"))
      logging.info("Dataset Loaded")

      os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

      df.to_csv(self.ingestion_config.raw_data_path, index=False)  

      train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
      logging.info("Raw data saved.")

      train_set.to_csv(self.ingestion_config.train_data_path, index=False)
      test_set.to_csv(self.ingestion_config.test_data_path, index=False)
      logging.info("Train test saved")

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path,
      )
    except Exception as e:
      logging.error("Error in data ingestion.")
      raise CustomException(e, sys)
    
if __name__ == "__main__":
  obj = DataIngestion()
  train_data, test_data = obj.initiate_data_ingestion()
  print("Data Ingestion Completed.")
      
  train_path = "artifacts/train.csv" 
  test_path = "artifacts/test.csv"

  transformer = DataTransformation()
  X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(
        train_path, test_path
    )

  print("Data Transformation Done.")  
