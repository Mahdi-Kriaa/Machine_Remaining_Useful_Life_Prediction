import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

import logging

def save_object(file_path, obj):
    # this function saves a python object in pkl format
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def find_best_hyperparameters(models_params, X_train, y_train, score):
    """
    this function return a dictionnary with models names as keys and 
    their best hyperparameters and corresponding scores as values
    """
    try :
        best_params = dict()
        for model_name in models_params.keys():
            model = models_params[model_name][0]
            model_params = models_params[model_name][1]
            grid_search = GridSearchCV(model, model_params,
                                        scoring=make_scorer(score), error_score="raise")
            grid_search.fit(X_train, y_train)
            best_params[model_name] = [grid_search.best_params_, grid_search.best_score_]
            print(f"{model_name} successfully completed")
        return best_params
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    # this function load a python object saved in pkl format
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def long_to_wide_form(data, n_in=1, n_out=1, dropnan=True, target=[], exep=[]):
    # this function transform a long form dataframe to wide form one
    n_vars = 1 if type(data) is list else data.shape[1]
    cols, namen = list(),list()
    vars = list(data.columns)
    data1 = data.drop(exep,axis=1)
    for e in exep :
      vars.remove(e)
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data1.shift(i))
        namen +=[('%s(t-%d)' %(s, i)) for s in vars]
        #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data1[target].shift(-i))
        if i == 0 :
          namen +=[(s+'(t)') for s in target]
        else :
          namen +=[(s+'(t+%d)' %(i)) for s in target]
    cols.append((data[exep]))
    namen += exep
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns=namen
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def transform_dataframe(df):
    # this function process each machine id data in a separate dataframe and then combine them in a single one
    
    df_list = []
    for machine_id in df["machineID"].unique():
        # keep only one machine id
        df_maintenance_temp = df[df["machineID"]==machine_id]

        # resample dataframe to daily frequency
        df_maintenance_temp = df_maintenance_temp.resample("d").agg({"volt":"mean", "rotate":"mean", "pressure":"mean", 
                                                                    "vibration":"mean", "model":"first", "age":"first", 
                                                                    "comp_count":"sum", "error_count":"sum", 
                                                                    "failure_component_count":"sum"})

        # remove rows with unknown RUL value
        failures_dates = df_maintenance_temp[df_maintenance_temp["failure_component_count"]!=0].index
        first_failure_date = failures_dates[0]
        last_failure_date = failures_dates[-1]
        df_maintenance_temp = df_maintenance_temp[(df_maintenance_temp.index>=first_failure_date) & (df_maintenance_temp.index<=last_failure_date)]

        # add RUL column to the dataframe
        rul_list = []
        j = 0
        failure_date = failures_dates[0] 
        for i in df_maintenance_temp.index:
            if df_maintenance_temp.loc[i, "failure_component_count"] != 0:
                rul_list.append(0)
                if j<(len(failures_dates)-1):
                    j += 1
                    failure_date = failures_dates[j]     
            else:
                rul_list.append((failure_date-i).days)
        df_maintenance_temp["RUL"] = rul_list

        df_maintenance_temp = long_to_wide_form(data=df_maintenance_temp, n_in=30, exep=["model", "age", "RUL"])
        
        df_list.append(df_maintenance_temp) # add dataframe
        df_processed = pd.concat(df_list) # combine dataframes

        return(df_processed)


