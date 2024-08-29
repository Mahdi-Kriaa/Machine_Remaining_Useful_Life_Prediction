import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path = "artifacts/python_objects/preprocessor.pkl"

class data_transformer:
    def __init__(self):
        self.data_transformation_config=data_transformation_config()

    def get_data_transformer_object(self, data_path, target_feature):
        # this function retunr a data transformer object
        try:
            df = pd.read_csv(data_path)
            num_cols = df.drop(columns=target_feature).select_dtypes("float").columns
            cat_cols = df.select_dtypes(exclude="float").columns
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())    
                ]
                )
            
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipelines", cat_pipeline, cat_cols)
                ],
                sparse_threshold=0
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path, target_feature):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object...")

            preprocessing_obj=self.get_data_transformer_object(train_data_path, target_feature)

            X_train = train_df.drop(columns=target_feature)
            y_train = train_df[target_feature]

            X_test = test_df.drop(columns=target_feature)
            y_test = test_df[target_feature]

            logging.info(
                f"Applying preprocessing object on training and testing dataframes."
            )

            transformed_X_train = preprocessing_obj.fit_transform(X_train)
            transformed_X_test = preprocessing_obj.transform(X_test)

            train_array = np.c_[transformed_X_train, np.array(y_train)]
            test_array = np.c_[transformed_X_test, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            