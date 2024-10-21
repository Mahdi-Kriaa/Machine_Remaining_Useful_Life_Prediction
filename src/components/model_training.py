import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

# data preprocessing
from src.components.data_transformation import data_transformer

# modeling
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.ops import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, BackupAndRestore
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam

@dataclass
class model_training_config:
    nn_model_file_path=os.path.join("artifacts/callbacks/nn_model_checkpoint.keras")
    backup_file_dir=os.path.join("artifacts/callbacks/backup")

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
            
            # define time steps and non time steps inputs
            time_steps_inputs = Input(shape=(30,7))
            non_time_steps_inputs = Input(shape=(5,))

            # define the lstm layers for the time steps inputs
            x = LSTM(64, activation="relu", return_sequences=True)(time_steps_inputs)
            x = LSTM(64, activation="relu", return_sequences=True)(x)
            x = LSTM(16, activation="relu")(x)
            x = Model(inputs=time_steps_inputs, outputs=x)

            # combine the outputs of the lstm layers outputs and the other inputs
            combined = concatenate([x.output, non_time_steps_inputs], axis=1)

            # define dense layers
            y = Dense(64, activation="relu")(combined)
            y = Dense(64, activation="relu")(y)
            y = Dense(4, activation="relu")(y)
            y = Dense(1, activation="relu")(y)

            # define the model
            nn_model = Model(inputs=[x.input, non_time_steps_inputs], outputs=y)

           # compile model
            nn_model.compile(loss="mse",
                        optimizer=Adam(learning_rate=0.0005),
                        metrics=["mae"])
            
            # define callbacks for the model
            model_checkpoint = ModelCheckpoint(
                filepath=self.model_trainer_config.nn_model_file_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True)
            backup = BackupAndRestore(backup_dir=self.model_trainer_config.backup_file_dir)
            model_callbacks=[backup, model_checkpoint]

            logging.info("The model training begin")
            
            # train model
            model_history = nn_model.fit(x=[time_steps_train_input, non_time_steps_train_input],
                    y=y_train,                         
                    batch_size=32,
                    epochs=700,
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
