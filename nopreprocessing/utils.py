import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, mean_absolute_percentage_error
)
from decimal import Decimal

def evaluate(y_true, y_pred, tasktype, verbose = 1):
    metrics = {}
    if tasktype == 1:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['precision'] = precision_score(y_true, y_pred, average = 'micro')
        metrics['recall'] = recall_score(y_true, y_pred, average = 'micro')

        if verbose == 1:
            print('Accuracy: ', metrics['accuracy'])
            print('F1 Score: ', metrics['f1'])
            print('Precision: ', metrics['precision'])
            print('Recall: ', metrics['recall'])
            print('Classification Report: ')
            print(classification_report(y_true, y_pred))
    else:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)

        if verbose == 1:
            print('Mean Squared Error: ', metrics['mse'])
            print('Root Mean Squared Error: ', metrics['rmse'])
            print('Mean Absolute Error: ', metrics['mae'])
            print('Mean Absolute Percentage Error: ', metrics['mape'])
            print('R2 Score: ', metrics['r2'])
    return metrics

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

        #plot the last 50 test data
        plt.figure(figsize = (30, 10))
        plt.plot(y_true_test[-50:].values, label = 'True')
        plt.plot(y_pred_test[-50:].values, label = 'Predicted')
        plt.xticks(np.arange(0, 50, step = 2))
        plt.legend()
        plt.title(
            f'{modelname} Prediction Result\n'+ testmetricshow
        )
        plt.xlabel('Index')
        plt.ylabel('Value')
        if savepath is not None:
            splitted = savepath.split('.')[0] + '_testlast50' + savepath.split('.')[1]
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