import pandas as pd
import numpy as np
import dill
import os
import sys
from src.logger import logging
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path,obj):
      try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path , exist_ok=True)
            
            with open(file_path,"wb") as file_obj:
                  dill.dump(obj,file_obj)
            logging.info(" pkl file created successfully ")
      except Exception as e:
            raise CustomException(e,sys)

def evaluate_model(x_train , y_train , x_test , y_test , models):
      try:
            logging.info("starting model training")
            
            report = {}
            
            for model_name , model in models.items() :
                  model.fit(x_train,y_train)
                  y_train_pred = model.predict(x_train)
                  y_test_pred = model.predict(x_test)
                  train_model_score = r2_score(y_train , y_train_pred)
                  test_model_score = r2_score(y_test , y_test_pred)
                  # report[f"{model_name}_train_score"] = train_model_score
                  # report[f"{model_name}_test_score"] = test_model_score
                  # logging.info(f"model traxined {model_name} with train accuracy {train_model_score} and test accuracy {test_model_score}")
                  report[model_name] = test_model_score
                  logging.info(f"{model_name} training completed with accuracy {test_model_score}")
            logging.info(report)
            return report
      except Exception as e:
            raise CustomException(e,sys)