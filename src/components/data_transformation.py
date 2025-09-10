import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
from src.exception import CustomException
from src.logger import logging


class DataTransformation:
  def __init__(self):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    self.preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")

  def get_data_transformation_object(self, numerical_cols, categorical_cols):
    try:
      num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
      ])  

      cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown='ignore',sparse_output=False))

      ])

      logging.info("Pipelines created for numerical and categorical features.")

      preprocessor = ColumnTransformer(
        transformers=[('num', num_pipeline, numerical_cols),
                      ('cat', cat_pipeline, categorical_cols)]
      )
      return preprocessor
    except Exception as e:
      raise CustomException(e, sys)

  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)

      logging.info("Read train and test data")

      target_column = "Churn"

      if "customerID" in train_df.columns:
                train_df = train_df.drop(columns=["customerID"])
      if "customerID" in test_df.columns:
                test_df = test_df.drop(columns=["customerID"])


      numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

      categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

      if target_column in numerical_cols:
        numerical_cols.remove(target_column)
      if target_column in categorical_cols:
        categorical_cols.remove(target_column)

      preprocessing_obj = self.get_data_transformation_object(numerical_cols, categorical_cols)    

      X_train = train_df.drop(columns=[target_column],axis=1)
      y_train = train_df[target_column]

      X_test = test_df.drop(columns=[target_column],axis=1)
      y_test = test_df[target_column]

      X_train = preprocessing_obj.fit_transform(X_train)
      X_test = preprocessing_obj.transform(X_test)

      logging.info("Applied Preprocessing.")

      os.makedirs(os.path.dirname(self.preprocessor_file_path), exist_ok=True)


      with open(self.preprocessor_file_path, 'wb') as f:
        pickle.dump(preprocessing_obj,f)

      return X_train, X_test, y_train, y_test, self.preprocessor_file_path
    except Exception as e:
      raise CustomException(e,sys)  