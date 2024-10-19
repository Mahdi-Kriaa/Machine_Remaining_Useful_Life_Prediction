import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

# data preprocessing
from src.components.data_transformation import data_transformer

# modeling
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Dropout, LSTM, Dense, TimeDistributed
from tensorflow.keras.ops import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, BackupAndRestore, EarlyStopping
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam

@dataclass
class model_training_config:
    trained_model_file_path=os.path.join("artifacts/python_objects/callbacks/trained_model.keras")
    backup_file_dir=os.path.join("artifacts/python_objects/callbacks/backup")

class model_trainer:
    def __init__(self):
        self.model_trainer_config=model_training_config()


    def initiate_model_trainer(self, train_data_path, val_data_path, test_data_path, target_feature):
        try:
            data_transformer_obj = data_transformer()
            train_array, val_array, test_array = data_transformer_obj.initiate_data_transformation(train_data_path, 
                                                                                        test_data_path,
                                                                                        val_data_path, 
                                                                                        target_feature)
            
            X_train,y_train,X_val,y_val,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                val_array[:,:-1],
                val_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # define two sets of inputs representing time steps and non time steps features
            # train inputs
            time_steps_train_input = X_train[:, :-5].reshape((-1,30,7))
            non_time_steps_train_input = X_train[:, -5:]

            # validation inputs
            time_steps_val_input = X_val[:, :-5].reshape((-1,30,7))
            non_time_steps_val_input = X_val[:, -5:]

            # test inputs
            time_steps_test_input = X_test[:, :-5].reshape((-1,30,7))
            non_time_steps_test_input = X_test[:, -5:]

            # define two sets of inputs
            time_steps_inputs = Input(shape=(30,7))
            non_time_steps_inputs = Input(shape=(5,))
            # define the first branch operating on the first input
            x1 = LSTM(100, activation="relu", return_sequences=True)(time_steps_inputs)
            x1 = LSTM(20, activation="relu")(x1)
            x1 = Model(inputs=time_steps_inputs, outputs=x1)
            # tdefine he second branch opreating on the second input
            x2 = Dense(64, activation="relu")(non_time_steps_inputs)
            x2 = Dense(32, activation="relu")(x2)
            x2 = Dense(4, activation="relu")(x2)
            x2 = Model(inputs=non_time_steps_inputs, outputs=x2)
            # combine the output of the two branches
            combined = concatenate([x1.output, x2.output], axis=1)
            # apply a FC layer and then a regression prediction on the ombined outputs
            x3 = Dense(2, activation="relu")(combined)
            x3 = Dense(1, activation="relu")(x3)
            # our model will accept the inputs of the two branches and
            # then output a single value
            model = Model(inputs=[x1.input, x2.input], outputs=x3)

           # compile model
            model.compile(loss="mse",
                        optimizer=Adam(learning_rate=0.0005),
                        metrics=["mae"])
            
            # define callbacks for the model
            model_checkpoint = ModelCheckpoint(
                filepath=self.model_trainer_config.trained_model_file_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True)
            backup = BackupAndRestore(backup_dir=self.model_trainer_config.backup_file_dir)
            model_callbacks=[backup, model_checkpoint]

            logging.info("The model training begin")
            
            # train model
            model_history = model.fit(x=[time_steps_train_input, non_time_steps_train_input],
                    y=y_train,                         
                    batch_size=32,
                    epochs=2,
                    validation_data=([time_steps_val_input, non_time_steps_val_input], y_val),
                    validation_batch_size=32,
                    callbacks=model_callbacks) 
            
            logging.info("The model training is completed")
                
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    train_data_path = "artifacts/data/training_data.csv"
    val_data_path = "artifacts/data/validation_data.csv"
    test_data_path = "artifacts/data/test_data.csv"
    model_trainer_obj = model_trainer()
    model_trainer_obj.initiate_model_trainer(train_data_path, val_data_path, 
                                                      test_data_path, "RUL")
