import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import data_transformer
from src.utils import save_object, find_best_hyperparameters

@dataclass
class model_training_config:
    trained_model_file_path=os.path.join("artifacts/python_objects/model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer_config=model_training_config()


    def initiate_model_trainer(self, train_data_path, test_data_path, target_feature):
        try:
            data_transformer_obj = data_transformer()
            train_array, test_array, _ = data_transformer_obj.initiate_data_transformation(train_data_path, test_data_path, target_feature)
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
                                                
            # define different parameters for models
            linear_reg_params = dict(positive=[True, False])
            knn_params = dict(n_neighbors=[2, 5, 20, 50, 100])
            decision_tree_params = dict(max_depth=[None, 5, 50, 200],
                                        min_samples_split=[2, 20, 50, 100])
    
            
             # define a dictionnary for models and their hyperparameters
            models_params = dict(
                   linear_reg=[LinearRegression(), linear_reg_params],
                   knn=[KNeighborsRegressor(), knn_params],
                   decision_tree=[DecisionTreeRegressor(), decision_tree_params],
            )
 
            best_params = find_best_hyperparameters(models_params, X_train, y_train)
            best_model_name = list(best_params.keys())[0]
            for model_name in best_params.keys():
                if best_params[model_name][1] > best_params[best_model_name][1]:
                    best_model_name = model_name

            if best_params[best_model_name][1] < 0.1:
                raise CustomException("Was not found a good model")
            logging.info(f"Best found model on both training and testing dataset")

            best_model = models_params[best_model_name][0].set_params(**best_params[best_model_name][0])
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            RMSE = mean_squared_error(y_test, predicted, squared=False)
            return RMSE
            



            
        except Exception as e:
            raise CustomException(e,sys)