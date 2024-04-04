import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, mean_absolute_percentage_error
)

def evaluate(y_true, y_pred, tasktype, verbose = 1):
    metrics = {}
    if tasktype == 1:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')
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
        