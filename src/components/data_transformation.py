import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# data preprocessing
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path = "artifacts/python_objects/preprocessor.pkl"

class data_transformer:
    def __init__(self):
        self.data_transformation_config = data_transformation_config()

    def get_data_transformer_object(self, train_data_path, target):
        # this function return a data transformer object
        try:
            train_data = pd.read_csv(train_data_path) # load training data
            X_train = train_data.drop(columns=target)
            num_cols = X_train.select_dtypes(float).columns # define numerical columns
            cat_cols = X_train.select_dtypes(object).columns # define categorical columns

            # create a column transformer for features
            preprocessor = ColumnTransformer((("num_transformer", StandardScaler(), num_cols),
                                                ("cat_transformer", OneHotEncoder(sparse_output=False), cat_cols))) 

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_data_path, val_data_path, test_data_path, target):

        try:
            logging.info("Data transformation begin")

            # load data
            train_data = pd.read_csv(train_data_path)
            val_data = pd.read_csv(val_data_path)
            test_data = pd.read_csv(test_data_path)

            # get the preprocessor object
            preprocessing_obj=self.get_data_transformer_object(train_data_path, target)

            # define features and target
            X_train = train_data.drop(columns=target)
            y_train = train_data[target]

            X_val = val_data.drop(columns=target)
            y_val = val_data[target]

            X_test = test_data.drop(columns=target)
            y_test = test_data[target]

            # transform data
            transformed_X_train = preprocessing_obj.fit_transform(X_train)
            transformed_X_val = preprocessing_obj.transform(X_val)
            transformed_X_test = preprocessing_obj.transform(X_test)

            # add the target column
            train_array = np.c_[transformed_X_train, np.array(y_train)]
            val_array = np.c_[transformed_X_val, np.array(y_val)]
            test_array = np.c_[transformed_X_test, np.array(y_test)]

            # save preprocessor object
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            logging.info(f"Preprocessing object is saved")
            return(train_array, val_array, test_array)

        except Exception as e:
            raise CustomException(e,sys)
        