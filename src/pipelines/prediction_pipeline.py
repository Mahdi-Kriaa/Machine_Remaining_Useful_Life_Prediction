import sys
import os
import pandas as pd
from src.exception import CustomException
from tensorflow.keras.models import load_model
from src.utils import load_object, long_to_wide_form

class prediction_pipeline:
    def __init__(self):
        pass

    def predict(self, machine_history):
        try:
            # get the model and preprocessor paths
            model_path=os.path.join("artifacts", "python_objects", "callbacks", "trained_model.keras")
            preprocessor_path=os.path.join("artifacts", "python_objects", "preprocessor.pkl")
            
            # load the model and the preprocessor
            model = load_model(model_path)
            preprocessor = load_object(file_path=preprocessor_path)
             
            transformed_df = long_to_wide_form(data=machine_history, n_in=30, exep=["model", "age"]) # transform dataframe to wide form 
            
            input_array = preprocessor.transform(transformed_df)
            
            time_steps_train_input = input_array[:, :-5].reshape((-1,30,7))
            non_time_steps_train_input = input_array[:, -5:]

            preds = model.predict([time_steps_train_input, non_time_steps_train_input]) # get prediction

            return preds
    
        except Exception as e:
            raise CustomException(e,sys)