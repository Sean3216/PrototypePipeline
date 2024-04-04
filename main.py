from xgboostutils.model import XGBoostModel
from prophetutils.model import ProphetModel

from utils.data import load_data, preprocessing
from utils.evaluate import evaluate, predict_and_evaluate
from utils.utilitycode import plot_regression_result, inverse_scale_target  

import pandas as pd
import argparse
import yaml
import copy
import os

#define availablemodels
#Trying to add DeepAR model
classification_mod = ['xgboost']
regression_mod = ['xgboost','prophet'] #add DeepAR here

#parse arguments
parser = argparse.ArgumentParser(description='Train and evaluate model')
parser.add_argument(
    '--config', 
    type=str, 
    default='config.yaml', 
    help='path to config file'
    )
args = parser.parse_args()
#load config file
with open(args.config, 'r', encoding = 'utf-8') as file:
    config = yaml.safe_load(file)

init_conf = config['init_config']
data_conf = config['data_config']
model_conf = config['model_config']

#set a variable that captures the name of the target column
target = data_conf['target']

# load data
rawtrain = load_data(data_conf['traindir'])
if data_conf['testdir'] is not None:
    rawtest = load_data(data_conf['testdir'])


def main():
    ###first force set cuda device
    if model_conf['cuda'] == True:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print('CUDA device forced to use GPU 0')
    ###Check chosen model
    if init_conf['tasktype'].lower() == 'classification':
        init_conf['tasktype'] = 1 #classification
        if init_conf['model'] not in classification_mod:
            raise ValueError('Model not available for classification. Please choose from: {}'.format(classification_mod))
    elif init_conf['tasktype'].lower() == 'regression':
        init_conf['tasktype'] = 0 #regression
        if init_conf['model'] not in regression_mod:
            raise ValueError('Model not available for regression. Please choose from: {}'.format(regression_mod))
    else:
        raise ValueError('Task type not recognized. Please choose either classification or regression')

    #Preprocessing should depend on whether the data is for what model [XGBoost, Prophet, or Neural Network]
    if init_conf['model'].lower() in ['xgboost', 'prophet']:
        print("Model chosen requires the input to be in dataframe or array format.")
        print("Setting data to be in dataframe format...")
        return_as_df = True
    if init_conf['model'].lower() == 'deepar':
        print("Model chosen requires the input to be in dataloader format.")
        print("Setting data to be in dataloader format...")
        return_as_df = False

    ###Preprocess the data
    (
        train, 
        val, 
        test, 
        scaler_y,
        train_date,
        val_date,
        test_date
    ) = preprocessing(rawtrain, 
                      target,
                      init_conf['tasktype'],
                      dropped_col=data_conf['drop_cols'],
                      train_val_test_ratio=data_conf['train_val_test_ratio'],
                      fw_steps=data_conf['fw_steps'],
                      n_hist=data_conf['n_hist'],
                      shuffle=data_conf['shuffle'],
                      apply_scaler=data_conf['apply_scaler'],
                      returnasdf=return_as_df,
                      date_col=data_conf['date_col'],
                      preprocessed = data_conf['preprocessed']                      
    )

    print("Data loaded successfully!")

    #if it is a dataframe, reset index of train, val, test and date series to avoid index mismatch in the future
    if return_as_df:
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
        print("Train shape: ", train.shape)
        print("Test shape: ", test.shape)
        print('Data indexes: ', train.index, val.index, test.index)

        #check if train_date or val_date or test_date is not None
        if train_date is not None:
            train_date = train_date.reset_index(drop=True)
            print('Train date shape: ', train_date.shape)
            print('Train date indexes: ', train_date.index)
        if val_date is not None:
            val_date = val_date.reset_index(drop=True)
            print('Val date shape: ', val_date.shape)
            print('Val date indexes: ', val_date.index)
        if test_date is not None:
            test_date = test_date.reset_index(drop=True)
            print('Test date shape: ', test_date.shape)
            print('Test date indexes: ', test_date.index)

    ###Select and train using model
    ##XGBoost
    print('Check train columns: ', train.columns)
    if init_conf['model'] == 'xgboost':
        #initialize model
        model = XGBoostModel(tasktype = init_conf['tasktype'],
                             params = model_conf['hyperparameter_defaults'],
                             model_path = init_conf['modelpath'],
                             load_model = model_conf['load_model'],
                             cuda = model_conf['cuda'], scalertarget = scaler_y)
        
        #check if the hyperparameter tuning field is filled
        if model_conf['hyperparameter_tuning'] is not None: #if hyperparameter tuning field is filled
            print('contents of hyperparameter tuning: ', model_conf['hyperparameter_tuning'])
            model.hyperparameter_tuning(train.drop(target, axis=1), train[target], val.drop(target, axis=1), val[target], model_conf['hyperparameter_tuning'])
        
        #train the model
        model.train(train.drop(target, axis=1), train[target])

        #predict on test data
        if test is not None:
            metrics = predict_and_evaluate(model, test, target, scaler_y, type = 'test')
        else:
            metrics = predict_and_evaluate(model, train, target, scaler_y, type = 'train')
        model.save_model()

    ##Prophet
    elif init_conf['model'] == 'prophet':
        #check if date column is given
        if data_conf['date_col'] is None:
            print('Date column is needed for Prophet model! No date column found.')
            print('Using dummy date column...')

        #initialize model
        model = ProphetModel(tasktype = init_conf['tasktype'],
                             params = model_conf['hyperparameter_defaults'],
                             model_path = init_conf['modelpath'],
                             load_model = model_conf['load_model'],
                             cuda = model_conf['cuda'], scalertarget = scaler_y,
                             train_date = train_date, val_date = val_date, test_date = test_date)
        
        #check if the hyperparameter tuning field is filled
        if model_conf['hyperparameter_tuning'] is not None:
            print('contents of hyperparameter tuning: ', model_conf['hyperparameter_tuning'])
            model.hyperparameter_tuning(train.drop(target, axis=1), train[target], val.drop(target, axis=1), val[target], model_conf['hyperparameter_tuning'])
        
        #train the model
        model.train(train.drop(target, axis=1), train[target])

        #predict on test data
        if test is not None:
            metrics = predict_and_evaluate(model, test, target, scaler_y, type = 'test')
        else:
            metrics = predict_and_evaluate(model, train, target, scaler_y, type = 'train')
        model.save_model()
    
    ###Plotting the regression result for regression case
    if init_conf['tasktype'] == 0:
        #obtain metrics for train, val, and test data
        if test is not None:
            train_metrics = predict_and_evaluate(model, train, target, scaler_y, type = 'train', verbose=0)
            val_metrics = predict_and_evaluate(model, val, target, scaler_y, type = 'val', verbose=0)
            test_metrics = copy.deepcopy(metrics)
        else:
            train_metrics = copy.deepcopy(metrics)
            val_metrics = predict_and_evaluate(val[target], model.predict(val.drop(target, axis=1), type = 'val'), init_conf['tasktype'], verbose=0)
            test_metrics = None

        #inverse scale target if data is not preprocessed
        if data_conf['preprocessed'] == False:
            ytrainrescaled = inverse_scale_target(train[target], scaler_y)
            yvalrescaled = inverse_scale_target(val[target], scaler_y)
            ytestrescaled = inverse_scale_target(test[target], scaler_y)  
        else:
            ytrainrescaled = train[target]
            yvalrescaled = val[target]
            ytestrescaled = test[target]

        plot_regression_result(
            ytrainrescaled,
            model.predict(train.drop(target, axis=1), type = 'train'),
            yvalrescaled,
            model.predict(val.drop(target, axis=1), type = 'val'),
            train_metrics,
            val_metrics,
            test_metrics,
            ytestrescaled,
            model.predict(test.drop(target, axis=1), type = 'test'),
            modelname = init_conf['model'],
            savepath = str(init_conf['model'])+ '_'+ target + '_regression.png'  
        )    

if __name__ == '__main__':
    main()
        
