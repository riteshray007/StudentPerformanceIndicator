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
                        "linear Regressor":LinearRegression(),
                        "ridge Regressor" :Ridge(),
                        "KNN Regressor" : KNeighborsRegressor(),
                        "Decision tree regressor": DecisionTreeRegressor(),
                        "Random forest regressor": RandomForestRegressor(),
                        "Adaboost regressor": AdaBoostRegressor(),
                        "Catboost regressor": CatBoostRegressor(verbose=False),
                        "XGB regressor" : XGBRegressor()
                  }
                  
                  model_report:dict=evaluate_model(x_train , y_train , x_test , y_test , models)
                  best_model_name = max(model_report, key=model_report.get)
                  best_model_score = model_report[best_model_name]
                  
                  best_model = models[best_model_name]
                  
                  if best_model_score < 0.5:
                        raise CustomException('Even the best score from all models is less then 50%')
                  logging.info(f"Best model is {best_model_name}")
                  
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
