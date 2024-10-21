import os
import sys

import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate

from src.exception import CustomException


def save_object(file_path, obj):
    # this function saves a python object in pkl format
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

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
        try:
            # keep only one machine id
            df_maintenance_temp = df[df["machineID"]==machine_id]

            # resample dataframe to daily frequency
            df_maintenance_temp = df_maintenance_temp.resample("d").agg({"volt":"mean", "rotate":"mean", "pressure":"mean", 
                                                                        "vibration":"mean", "model":"first", "age":"first", 
                                                                        "comp_count":"sum", "error_count":"sum", 
                                                                        "failure_component_count":"sum"})

            # define failures dates
            failures_dates = df_maintenance_temp[df_maintenance_temp["failure_component_count"]!=0].index

            # remove rows with unknown remaining useful life
            df_maintenance_temp = df_maintenance_temp[df_maintenance_temp.index<=failures_dates[-1]]

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
            df_transformed = pd.concat(df_list) # combine dataframes
        except: pass
    return(df_transformed)

def plot_history(history, training_metric, validation_metric, figsize=(8,8)):
        # this function display the training and validation metrics

        # get training and validation metric
        train_metric = history.history[training_metric]
        val_metric = history.history[validation_metric]

        # get training and validation loss
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # plot training and validation mae
        plt.figure(figsize=figsize)
        plt.subplot(2, 1, 1)
        plt.plot(train_metric, label=training_metric)
        plt.plot(val_metric, label=validation_metric)
        plt.legend()
        plt.ylabel("MAE")

        # plot training and validation loss
        plt.subplot(2, 1, 2)
        plt.plot(train_loss, label="loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

def regression_metrics(model, X_train, X_test, y_train, y_test):
    # this function calculate the regression model metrics
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    tab = tabulate([["Metric", "Training Set", "Test Set"],
                  ["r2", r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)],
                  ["MSE", mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)],
                  ["MAE", mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)],
                  ["RMSE", mean_squared_error(y_train, y_train_pred, squared=False), mean_squared_error(y_test, y_test_pred, squared=False)]],
                headers='firstrow', numalign="left")
    return tab
