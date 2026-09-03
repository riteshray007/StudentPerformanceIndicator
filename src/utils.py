import pandas as pd
import numpy as np
import dill
import os
import sys
from src.logger import logging
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
      try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path , exist_ok=True)
            
            with open(file_path,"wb") as file_obj:
                  dill.dump(obj,file_obj)
            logging.info(" pkl file created successfully ")
      except Exception as e:
            raise CustomException(e,sys)

def evaluate_model(x_train , y_train , x_test , y_test , models , param ):
      try:
            logging.info("starting model training")
            
            report = {
                  "best_model_score": float("-inf"),
                  "best_model": None,
            }
            
            for model_name , model in models.items() :
                  
                  logging.info(f" searching best params for model - {model_name} ")
                  para=param[model_name]
                  gs = GridSearchCV(model , para , cv=3)
                  gs.fit(x_train,y_train)
                  logging.info(f" best params for {model_name} are - {gs.best_params_} ")
                  # logging.info(f" best estimator are - {gs.best_estimator_}")
                  # model.set_params(**gs.best_params_)
                  # model.fit(x_train,y_train)
                  current_score = gs.best_score_
                  current_model = gs.best_estimator_                  
                  # logging.info(f"model traxined {model_name} with train accuracy {current_score} and test accuracy {test_model_score}")
                  if( current_score > report["best_model_score"] ):
                        report["best_model_score"] = current_score
                        report["best_model"] = current_model
                  logging.info(f"{current_model} training completed with accuracy {current_score}")
            logging.info(report)
            return report
      except Exception as e:
            raise CustomException(e,sys)