import sys
import os

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.ensemble import ( AdaBoostRegressor , GradientBoostingRegressor , RandomForestRegressor )
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
      model_obj_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
      def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()
            
      def initiate_model_trainer(self,train_array,test_array):
            try:
                  logging.info("split train and test data")
                  x_train,y_train,x_test,y_test = (
                        train_array[:,:-1],
                        train_array[:,-1],
                        test_array[:,:-1],
                        test_array[:,-1]                        
                  )
                  models = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(),
                        "KNN Regressor": KNeighborsRegressor(),
                        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                        "Random Forest Regressor": RandomForestRegressor(random_state=42),
                        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                        "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                        "CatBoost Regressor": CatBoostRegressor(verbose=False, random_state=42,allow_writing_files=False),
                        "XGB Regressor": XGBRegressor(random_state=42),
                  }
                  
                  params = {
                        "Linear Regression": {},

                        "Ridge Regression": {
                              "alpha": [0.01, 0.1, 1, 10, 100]
                        },

                        "KNN Regressor": {
                              "n_neighbors": [3, 5, 7, 9, 11],
                              "weights": ["uniform", "distance"],
                              "p": [1, 2]
                        },

                        "Decision Tree Regressor": {
                              "criterion": ["squared_error", "absolute_error", "poisson"],
                              "max_depth": [None, 5, 10, 15],
                              "min_samples_split": [2, 5, 10],
                              "min_samples_leaf": [1, 2, 4]
                        },
                        
                        "Random Forest Regressor": {
                              "n_estimators": [100, 200, 300],
                              "max_depth": [None, 10, 20],
                              "min_samples_split": [2, 5],
                              "min_samples_leaf": [1, 2],
                              "max_features": ["sqrt", "log2"]
                        },

                        "Gradient Boosting Regressor": {
                              "n_estimators": [100, 200],
                              "learning_rate": [0.01, 0.05, 0.1],
                              "max_depth": [3, 5],
                              "subsample": [0.8, 1.0]
                        },

                        "AdaBoost Regressor": {
                              "n_estimators": [50, 100, 200],
                              "learning_rate": [0.01, 0.05, 0.1, 0.5],
                              "loss": ["linear", "square", "exponential"]
                        },

                        "CatBoost Regressor": {
                              "iterations": [200, 500],
                              "depth": [4, 6, 8],
                              "learning_rate": [0.03, 0.05, 0.1],
                              "l2_leaf_reg": [1, 3, 5]
                        },

                        "XGB Regressor": {
                              "n_estimators": [200, 500],
                              "learning_rate": [0.03, 0.05, 0.1],
                              "max_depth": [3, 5, 7],
                              "subsample": [0.8, 1.0],
                              "colsample_bytree": [0.8, 1.0]
                        }
                  }      
                  
                  model_report:dict=evaluate_model(x_train , y_train , x_test , y_test , models , params )
                  # best_model_name = max(model_report, key=model_report.get)
                  # best_model_score = model_report[best_model_name]
                  
                  best_model = model_report["best_model"]
                  
                  
                  if model_report["best_model_score"] < 0.5:
                        raise CustomException('Even the best score from all models is less then 50%')
                  logging.info(f"Best model is {best_model}")
                  
                  save_object(
                        file_path=self.model_trainer_config.model_obj_file_path,
                        obj=best_model
                  )
                  
                  predicted = best_model.predict(x_test)
                  return r2_score(y_test,predicted)
                  
            except Exception as e :
                  raise CustomException(e,sys)



# class ModelTrainer:
#       def __init__(self):
#             self.model_trainer_exec = self.modelTrainerExec()
            
#       def modelTrainerExec(self):
#             logging.info('entered model trainer ' )
      
# if __name__=="__main__":
#       ModelTrainer()
