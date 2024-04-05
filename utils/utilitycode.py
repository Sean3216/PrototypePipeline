import pandas as pd
import numpy as np
import torch
import yaml
from hyperopt import hp
import matplotlib.pyplot as plt
from decimal import Decimal

def label_checking(data, label):
    if label not in data.columns or label is None:
        raise ValueError('Label not found in data columns. Please provide the label column name')
    
def inverse_scale_target(y, scaler_y):
    #check if y is a pandas series
    if isinstance(y, pd.Series):
        y = pd.Series(scaler_y.inverse_transform(y.values.reshape(-1, 1)).flatten())
    #check if y is torch tensor
    elif isinstance(y, torch.Tensor):
        y = pd.Series(scaler_y.inverse_transform(y.detach().numpy().reshape(-1, 1)).flatten())
    #check if y is array
    elif isinstance(y, np.ndarray):
        y = pd.Series(scaler_y.inverse_transform(y.reshape(-1, 1)).flatten())
    return y
    
def check_and_sort_ts(data, date_col = None):
    #check if the data index is datetime
    if isinstance(data.index, pd.DatetimeIndex):
        date_col = data.index.name
    else:
        if date_col is None:
            raise ValueError('date_col must be provided for timeseries data')
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.set_index(date_col)
    #sort data by date
    data = data.sort_index()
    return data, date_col

def convert_to_hp_space(yaml_data):
    search_space = {}
    for param_spec in yaml_data:
        name = param_spec['name']
        dist_type = param_spec['type']
        if dist_type == 'randint':
            search_space[name] = hp.randint(name, param_spec['bounds'][0], param_spec['bounds'][1])
        elif dist_type == 'uniform':
            search_space[name] = hp.uniform(name, param_spec['bounds'][0], param_spec['bounds'][1])
        elif dist_type == 'loguniform':
            search_space[name] = hp.loguniform(name, param_spec['bounds'][0], param_spec['bounds'][1])
        elif dist_type == 'choice':
            search_space[name] = hp.choice(name, param_spec['values'])
        # Add more conditions for other distribution types as needed
    return search_space

def plot_regression_result(y_true_train, y_pred_train,
                           y_true_val, y_pred_val, 
                           metricstrain, metricsval, 
                           metricstest = None, y_true_test = None, y_pred_test = None,
                           modelname = '', savepath = None):
    if metricstest is not None and y_true_test is None:
        raise ValueError('y_true_test must be provided if metricstest is provided')
    elif metricstest is None and y_true_test is not None:
        raise ValueError('metricstest must be provided if y_true_test is provided')
    elif metricstest is not None and y_pred_test is None:
        raise ValueError('y_pred_test must be provided if metricstest is provided')
    elif metricstest is None and y_pred_test is not None:
        raise ValueError('metricstest must be provided if y_pred_test is provided')
    elif metricstest is not None and y_true_test is not None and y_pred_test is not None:
        #reset the index of y_true_train and y_pred_train
        y_true_train = y_true_train.reset_index(drop = True)
        y_pred_train = pd.Series(y_pred_train)
        y_pred_train = y_pred_train.reset_index(drop = True)

        y_true_val = y_true_val.reset_index(drop = True)
        y_pred_val = pd.Series(y_pred_val)
        y_pred_val = y_pred_val.reset_index(drop = True)
        
        y_true_test = y_true_test.reset_index(drop = True)
        y_pred_test = pd.Series(y_pred_test)
        y_pred_test = y_pred_test.reset_index(drop = True)

        #first check whether all y_true and all y_pred does not have missing values
        if y_true_train.isnull().sum() > 0 or y_true_val.isnull().sum() > 0 or y_true_test.isnull().sum() > 0:
            raise ValueError('y_true_train, y_true_val, and y_true_test must not have missing values')
        if y_pred_train.isnull().sum() > 0 or y_pred_val.isnull().sum() > 0 or y_pred_test.isnull().sum() > 0:
            raise ValueError('y_pred_train, y_pred_val, and y_pred_test must not have missing values')

        #make the index of y_true_val and y_pred_val to be after the last index of y_true_train and y_pred_train
        y_true_val.index = y_true_val.index + len(y_true_train) 
        y_pred_val.index = y_pred_val.index + len(y_pred_train)
        #make the index of y_true_test and y_pred_test to be after the last index of y_true_val and y_pred_val
        y_true_test.index = y_true_test.index + len(y_true_val)
        y_pred_test.index = y_pred_test.index + len(y_pred_val)

        val_start_index = len(y_true_train)
        test_start_index = len(y_true_train) + len(y_true_val)

        #check whether the indexes of 3 y_true series are unique
        if y_true_train.index.duplicated().sum() > 0 or y_true_val.index.duplicated().sum() > 0 or y_true_test.index.duplicated().sum() > 0:
            raise ValueError('y_true_train, y_true_val, and y_true_test must have unique indexes')
        #check whether the indexes of 3 y_pred series are unique
        if y_pred_train.index.duplicated().sum() > 0 or y_pred_val.index.duplicated().sum() > 0 or y_pred_test.index.duplicated().sum() > 0:
            raise ValueError('y_pred_train, y_pred_val, and y_pred_test must have unique indexes')

        #append y_true series
        y_true = y_true_train._append(y_true_val)._append(y_true_test).reset_index(drop = True)
        #append y_pred series
        y_pred = y_pred_train._append(y_pred_val)._append(y_pred_test).reset_index(drop = True)

        #convert all the values inside metrics train, val, and test to np.float64
        metricstrain = {key: round(Decimal(float(value)),3) for key, value in metricstrain.items()}
        metricsval = {key: round(Decimal(float(value)),3) for key, value in metricsval.items()}
        metricstest = {key: round(Decimal(float(value)),3) for key, value in metricstest.items()}

        print(metricstrain)
        print(metricsval)
        print(metricstest)
        trainmetricshow = f"Metrics Train| MSE: {metricstrain['mse']}, RMSE: {metricstrain['rmse']}, MAE: {metricstrain['mae']}, R^2: {metricstrain['r2']}, MAPE: {metricstrain['mape']}"
        valmetricshow = f"Metrics Validation| MSE: {metricsval['mse']}, RMSE: {metricsval['rmse']}, MAE: {metricsval['mae']}, R^2: {metricsval['r2']}, MAPE: {metricsval['mape']}"
        testmetricshow = f"Metrics Test| MSE: {metricstest['mse']}, RMSE: {metricstest['rmse']}, MAE: {metricstest['mae']}, R^2: {metricstest['r2']}, MAPE: {metricstest['mape']}"
    
        #plot a line plot for all the train, val, and test
        plt.figure(figsize = (30, 10))
        plt.plot(y_true, label = 'True')
        plt.plot(y_pred, label = 'Predicted')
        #plot axvline with "--" linestyle for validation and different linestyle for test
        plt.axvline(x = val_start_index, color = 'r', linestyle = '--', label = 'Validation Start')
        plt.axvline(x = test_start_index, color = 'g', linestyle = ':', label = 'Test Start')
        plt.legend()
        plt.title(
            f'{modelname} Prediction Result\n'+ trainmetricshow +'\n'+ valmetricshow +'\n'+ testmetricshow
            )
        plt.xlabel('Index')
        plt.ylabel('Value')
        if savepath is not None:
            splitted = savepath.split('.')[0] + '_full' + savepath.split('.')[1]
            plt.savefig(splitted)
        plt.show()

        #plot only the test data
        plt.figure(figsize = (30, 10))
        plt.plot(y_true_test, label = 'True')
        plt.plot(y_pred_test, label = 'Predicted')
        plt.legend()
        plt.title(
            f'{modelname} Prediction Result\n'+ testmetricshow
        )
        plt.xlabel('Index')
        plt.ylabel('Value')
        if savepath is not None:
            splitted = savepath.split('.')[0] + '_test' + savepath.split('.')[1]
            plt.savefig(splitted)
        plt.show()

        #plot the first 500 test data
        plt.figure(figsize = (30, 10))
        plt.plot(y_true_test[:500].values, label = 'True')
        plt.plot(y_pred_test[:500].values, label = 'Predicted')
        plt.legend()
        plt.title(
            f'{modelname} Prediction Result\n'+ testmetricshow
        )
        plt.xlabel('Index')
        plt.ylabel('Value')
        if savepath is not None:
            splitted = savepath.split('.')[0] + '_test500' + savepath.split('.')[1]
            plt.savefig(splitted)
        plt.show()
    else:
        #reset the index of y_true_train and y_pred_train
        y_true_train = y_true_train.reset_index(drop = True)
        y_pred_train = pd.Series(y_pred_train)
        y_pred_train = y_pred_train.reset_index(drop = True)
        y_true_val = y_true_val.reset_index(drop = True)
        y_pred_val = pd.Series(y_pred_val)
        y_pred_val = y_pred_val.reset_index(drop = True)

        #make the index of y_true_val and y_pred_val to be after the last index of y_true_train and y_pred_train
        y_true_val.index = y_true_val.index + len(y_true_train)
        y_pred_val.index = y_pred_val.index + len(y_pred_train)

        val_start_index = len(y_true_train)

        #combine all the data
        y_true = pd.concat([y_true_train, y_true_val])
        y_pred = pd.concat([y_pred_train, y_pred_val])

        #plot a line plot for all the train, val, and test
        plt.figure(figsize = (20, 10))
        plt.plot(y_true, label = 'True')
        plt.plot(y_pred, label = 'Predicted')
        #plot axvline with "--" linestyle for validation and different linestyle for test
        plt.axvline(x = val_start_index, color = 'r', linestyle = '--', label = 'Validation Start')
        plt.legend()
        plt.title(
            f'{modelname} Prediction Result\n'+ trainmetricshow +'\n'+ valmetricshow
            )
        plt.xlabel('Index')
        plt.ylabel('Value')

        #save the plot if savepath is provided
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()


