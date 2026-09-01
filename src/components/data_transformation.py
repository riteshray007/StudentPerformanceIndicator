import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTranformationConfig:
      preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
      

class DataTransformation:
      def __init__(self):
            self.data_transformation_config = DataTranformationConfig()
            
      numerical_columns=['writing_score','reading_score']
      categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
      target_column = ['math_score']
      
      def get_data_transformer_object(self):
            try:

                  num_pipeline = Pipeline(
                        steps=[
                              ( "imputer", SimpleImputer(strategy='median') ),
                              ('scaler' , StandardScaler() )
                        ]
                  )
                  
                  logging.info("standarzation of numerical columns completed ")
                  
                  cat_pipeline = Pipeline(
                        steps=[
                              ("imputer" , SimpleImputer(strategy='most_frequent') ),
                              ("one_hot_encoder" , OneHotEncoder(handle_unknown='ignore')  ),
                              ("scaler" , StandardScaler(with_mean=False) )
                        ]
                  )
                  
                  logging.info("encoding of categorical columns completed ")
                  
                  preprocessor = ColumnTransformer(
                        [
                              ("num_pipeline" , num_pipeline , self.numerical_columns ),
                              ("cat_pipeline" , cat_pipeline , self.categorical_columns )
                        ]
                  )

                  logging.info("combined categorical ans numerical pipeline together")
                  
                  return preprocessor                  
            except Exception as e :
                  raise CustomException(e,sys)
            
      def initiate_data_transformation(self,train_path,test_path):
            try:
                  train_df = pd.read_csv(train_path)
                  test_df = pd.read_csv(test_path)
                  
                  logging.info('Reading train and test data completed')
                  
                  logging.info('obtainnig preprocessing object')
                  
                  preprocessing_obj =  self.get_data_transformer_object()
                  
                  input_feature_train_df = train_df.drop(columns=self.target_column , axis=1)
                  target_feature_train_df = train_df[self.target_column]
                  
                  input_feature_test_df = test_df.drop(columns=self.target_column , axis=1 )
                  target_feature_test_df = test_df[self.target_column]
                  
                  logging.info(
                        "applying preprocessor obj on train and test dataframe"
                  )
                  
                  processed_train_df = preprocessing_obj.fit_transform(input_feature_train_df)
                  processed_test_df = preprocessing_obj.transform(input_feature_test_df)

                  train_arr = np.c_[
                        processed_train_df , np.array(target_feature_train_df)
                  ]
                  test_arr = np.c_[
                        processed_test_df , np.array(target_feature_test_df)
                  ]

                  logging.info('saved preprocessed objects ')
                  
                  save_object(
                        file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj
                  )
                  
                  return (
                        train_arr,
                        test_arr,
                        self.data_transformation_config.preprocessor_obj_file_path
                  )
                                            
            except Exception as e :
                  raise CustomException(e,sys)