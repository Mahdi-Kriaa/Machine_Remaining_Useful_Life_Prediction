import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

# data processing
from sklearn.model_selection import train_test_split
from src.utils import long_to_wide_form, transform_dataframe
from dataclasses import dataclass

@dataclass
class data_ingestion_config:
    telemetry_data_path = "raw_data/PdM_telemetry.csv"
    errors_data_path = "raw_data/PdM_errors.csv"
    failures_data_path = "raw_data/PdM_failures.csv"
    machines_data_path = "raw_data/PdM_machines.csv"
    component_data_path = "raw_data/PdM_maint.csv"
    training_data_path: str = "artifacts/data/training_data.csv"
    validation_data_path: str = "artifacts/data/validation_data.csv"
    test_data_path: str = "artifacts/data/test_data.csv"

class data_ingestor:
    def __init__(self):
        self.ingestion_config=data_ingestion_config()

    def initiate_data_ingestor(self):
        logging.info("Data ingestion begin")
        try:
            # load data
            df_telemetry = pd.read_csv(self.ingestion_config.telemetry_data_path)
            df_errors = pd.read_csv(self.ingestion_config.errors_data_path)
            df_failures = pd.read_csv(self.ingestion_config.failures_data_path)
            df_machines = pd.read_csv(self.ingestion_config.machines_data_path)
            df_components = pd.read_csv(self.ingestion_config.component_data_path)

            # creates training and test data directories
            os.makedirs(os.path.dirname(self.ingestion_config.training_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.validation_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)

            # create an error count column for each datetime and machine id pair
            df_components = df_components.groupby(by=["datetime", "machineID"], as_index=False).count()
            df_components.rename(columns={"comp": "comp_count"}, inplace=True)
            
            # create an error count column for each datetime and machine id pair
            df_errors = df_errors.groupby(by=["datetime", "machineID"], as_index=False).count()
            df_errors.rename(columns={"errorID": "error_count"}, inplace=True)
                        
            # create an failures comonent count column for each datetime and machine id pair
            df_failures = df_failures.groupby(by=["datetime", "machineID"], as_index=False).count()
            df_failures.rename(columns={"failure": "failure_component_count"}, inplace=True)

            # join dataframes
            df_maintenance = df_telemetry.merge(df_machines, on="machineID", how="left").merge(df_components, 
                                                on=["datetime", "machineID"], how="left").merge(df_errors, 
                                                on=["datetime", "machineID"], how="left").merge(df_failures, 
                                                on=["datetime", "machineID"], how="left")

            
            df_maintenance.fillna(0, inplace=True) # replace nan values by 0
            df_maintenance["age"] = df_maintenance["age"].astype(float) # change the age column type to float
            df_maintenance["datetime"] = pd.to_datetime(df_maintenance["datetime"]) # change datetime column type to datetime
            df_maintenance.set_index("datetime", inplace=True) # set datetime column as index
            df_transformed = transform_dataframe(df_maintenance) # transform dataframe
            df_transformed.reset_index(inplace=True, drop=True) # reset index

            # splitting dataset into training, validation and testing data
            training_data, temp_data = train_test_split(df_maintenance, test_size=0.3, random_state=42)
            validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

           
            # save train and test data into csv files
            training_data.to_csv(self.ingestion_config.training_data_path,index=False,header=True)
            validation_data.to_csv(self.ingestion_config.validation_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.training_data_path,
                self.ingestion_config.training_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=data_ingestor()
    train_data_path, validation_data_path, test_data_path = obj.initiate_data_ingestor()