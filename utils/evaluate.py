import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, mean_absolute_percentage_error
)
from utils.utilitycode import inverse_scale_target

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

def predict_and_evaluate(model, data, target, scaler_y = None, type = 'train', verbose = 1, NN = False):
    if NN:
        y_pred = model.predict(type = type)
        if type.lower() == 'train':
            print('No test data. Predicting on train data...')
            print('Obtaining result on train data...')
            if scaler_y is not None:
                ytrainrescaled = inverse_scale_target(data[target], scaler_y)
                metrics = evaluate(ytrainrescaled, y_pred, model.tasktype, verbose = verbose)
            else:
                metrics = evaluate(data[target], y_pred, model.tasktype, verbose = verbose)
        elif type.lower() == 'val':
            print('Obtaining result on validation data...')
            if scaler_y is not None:
                yvalrescaled = inverse_scale_target(data[target], scaler_y)
                metrics = evaluate(yvalrescaled, y_pred, model.tasktype, verbose = verbose)
            else:
                metrics = evaluate(data[target], y_pred, model.tasktype, verbose = verbose)
        elif type.lower() == 'test':
            print('Obtaining result on test data...')
            if scaler_y is not None:
                ytestrescaled = inverse_scale_target(data[target], scaler_y)
                metrics = evaluate(ytestrescaled, y_pred, model.tasktype, verbose = verbose)
            else:
                metrics = evaluate(data[target], y_pred, model.tasktype, verbose = verbose)
        else:
            print('Invalid type. Please choose between train, val, or test.')
        return metrics
    else:
        y_pred = model.predict(data.drop(target, axis=1), type = type)
        if type.lower() == 'train':
            print('No test data. Predicting on train data...')
            print('Obtaining result on train data...')
            if scaler_y is not None:
                ytrainrescaled = inverse_scale_target(data[target], scaler_y)
                metrics = evaluate(ytrainrescaled, y_pred, model.tasktype, verbose = verbose)
            else:
                metrics = evaluate(data[target], y_pred, model.tasktype, verbose = verbose)
        elif type.lower() == 'val':
            print('Obtaining result on validation data...')
            if scaler_y is not None:
                yvalrescaled = inverse_scale_target(data[target], scaler_y)
                metrics = evaluate(yvalrescaled, y_pred, model.tasktype, verbose = verbose)
            else:
                metrics = evaluate(data[target], y_pred, model.tasktype, verbose = verbose)
        elif type.lower() == 'test':
            print('Obtaining result on test data...')
            if scaler_y is not None:
                ytestrescaled = inverse_scale_target(data[target], scaler_y)
                metrics = evaluate(ytestrescaled, y_pred, model.tasktype, verbose = verbose)
            else:
                metrics = evaluate(data[target], y_pred, model.tasktype, verbose = verbose)
        else:
            print('Invalid type. Please choose between train, val, or test.')
        return metrics
        