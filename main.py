from xgboostutils.model import XGBoostModel
from prophetutils.model import ProphetModel

from utils.data import load_data, preprocessing
from utils.evaluate import evaluate
from utils.utilitycode import plot_regression_result

import pandas as pd
import argparse
import yaml
import copy

#define availablemodels
classification_mod = ['xgboost']
regression_mod = ['xgboost','prophet']

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

# load data
rawtrain = load_data(data_conf['traindir'])
if data_conf['testdir'] is not None:
    rawtest = load_data(data_conf['testdir'])

#Function below will check first whether test data is provided or not.
#If not, it will split the train data into train and test data
def main():
    ###check chosen model
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

    (
        train, 
        val, 
        test, 
        input_size, 
        out, 
        scaler_x, 
        scaler_y, 
        label_mapping,
        train_date,
        val_date,
        test_date
    ) = preprocessing(rawtrain, 
                      data_conf['target'],
                      init_conf['tasktype'],
                      dropped_col=data_conf['drop_cols'],
                      train_val_test_ratio=data_conf['train_val_test_ratio'],
                      fw_steps=data_conf['fw_steps'],
                      n_hist=data_conf['n_hist'],
                      shuffle=data_conf['shuffle'],
                      apply_scaler=data_conf['apply_scaler'],
                      returnasdf=data_conf['return_as_df'],
                      date_col=data_conf['date_col'],
                      preprocessed = data_conf['preprocessed']                      
    )

    print("Data loaded successfully!")
    if data_conf['return_as_df']:
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

    
    if type(train) == pd.DataFrame or type(val) == pd.DataFrame or type(test) == pd.DataFrame:
        target = data_conf['target']
    ###select model
    ##XGBoost
    if init_conf['model'] == 'xgboost':
        #first force set cuda device
        if model_conf['cuda'] == True:
            import os
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print('CUDA device forced to use GPU 0')
        #need to drop date column if it exists because not supported
        if data_conf['date_col'] is not None:
            train = train.drop(data_conf['date_col'], axis=1)
            val = val.drop(data_conf['date_col'], axis=1)
            if test is not None:
                test = test.drop(data_conf['date_col'], axis=1)
        model = XGBoostModel(tasktype = init_conf['tasktype'],
                             params = model_conf['hyperparameter_defaults'],
                             model_path = init_conf['modelpath'],
                             load_model = model_conf['load_model'],
                             cuda = model_conf['cuda'], scalertarget = scaler_y)
        if model_conf['hyperparameter_tuning'] is not None: #if hyperparameter tuning field is filled
            print('contents of hyperparameter tuning: ', model_conf['hyperparameter_tuning'])
            model.hyperparameter_tuning(train.drop(target, axis=1), train[target], val.drop(target, axis=1), val[target], model_conf['hyperparameter_tuning'])
        model.train(train.drop(target, axis=1), train[target])

        if test is not None:
            y_pred = model.predict(test.drop(target, axis=1)) #note: prediction result does not need to be rescaled because the decision is inside model.predict
        else:
            print('No test data. Predicting on train data...')
            y_pred = model.predict(train.drop(target, axis=1)) #note: prediction result does not need to be rescaled because the decision is inside model.predict
        
        print('Obtaining result on test data...')
        if init_conf['tasktype'] == 0:
            if scaler_y is not None:
                ytestrescaled = scaler_y.inverse_transform(test[target].values.reshape(-1, 1))
                ytestrescaled = pd.Series(ytestrescaled.flatten())
            else:
                ytestrescaled = test[target]
        metrics = evaluate(ytestrescaled, y_pred, init_conf['tasktype']) #save metrics for now in case we need it later
        model.save_model()

    ##Prophet
    elif init_conf['model'] == 'prophet':
        #check if date column is given
        if data_conf['date_col'] is None:
            print('Date column is needed for Prophet model! No date column found.')
            print('Using dummy date column...')
        model = ProphetModel(tasktype = init_conf['tasktype'],
                             params = model_conf['hyperparameter_defaults'],
                             model_path = init_conf['modelpath'],
                             load_model = model_conf['load_model'],
                             cuda = model_conf['cuda'], scalertarget = scaler_y,
                             train_date = train_date, val_date = val_date, test_date = test_date)
        if model_conf['hyperparameter_tuning'] is not None:
            print('contents of hyperparameter tuning: ', model_conf['hyperparameter_tuning'])
            model.hyperparameter_tuning(train.drop(target, axis=1), train[target], val.drop(target, axis=1), val[target], model_conf['hyperparameter_tuning'])
        model.train(train.drop(target, axis=1), train[target])

        if test is not None:
            y_pred = model.predict(test.drop(target, axis=1), type = 'test') #note: prediction result does not need to be rescaled because the decision is inside model.predict
            print('Obtaining result on test data...')
            if scaler_y is not None:
                ytestrescaled = scaler_y.inverse_transform(test[target].values.reshape(-1, 1))
                ytestrescaled = pd.Series(ytestrescaled.flatten())
            else:
                ytestrescaled = test[target]
            metrics = evaluate(ytestrescaled, y_pred, init_conf['tasktype'])

        else:
            print('No test data. Predicting on train data...')
            y_pred = model.predict(train.drop(target, axis=1), type = 'train') #note: prediction result does not need to be rescaled because the decision is inside model.predict
            print('Obtaining result on train data...')
            if scaler_y is not None:
                ytrainrescaled = scaler_y.inverse_transform(train[target].values.reshape(-1, 1))
                ytrainrescaled = pd.Series(ytrainrescaled.flatten())
            else:
                ytrainrescaled = train[target]
            metrics = evaluate(ytrainrescaled, y_pred, init_conf['tasktype'])
        model.save_model()
    
    if init_conf['tasktype'] == 0:
        if data_conf['preprocessed'] == False:
            ytrainrescaled = scaler_y.inverse_transform(train[target].values.reshape(-1, 1))
            yvalrescaled = scaler_y.inverse_transform(val[target].values.reshape(-1, 1))
            ytestrescaled = scaler_y.inverse_transform(test[target].values.reshape(-1, 1))  

            ytrainrescaled = pd.Series(ytrainrescaled.flatten())
            yvalrescaled = pd.Series(yvalrescaled.flatten())
            ytestrescaled = pd.Series(ytestrescaled.flatten())
        else:
            ytrainrescaled = train[target]
            yvalrescaled = val[target]
            ytestrescaled = test[target]

        if test is not None:
            if init_conf['model'] == 'prophet':
                train_metrics = evaluate(
                    train[target],
                    model.predict(train.drop(target, axis=1), type = 'train'),
                    init_conf['tasktype'],
                    verbose=0)
                val_metrics = evaluate(
                    val[target],
                    model.predict(val.drop(target, axis=1), type = 'val'),
                    init_conf['tasktype'],
                    verbose=0)
                test_metrics = copy.deepcopy(metrics)
            elif init_conf['model'] == 'xgboost':
                train_metrics = evaluate(
                    train[target],
                    model.predict(train.drop(target, axis=1)),
                    init_conf['tasktype'],
                    verbose=0)
                val_metrics = evaluate(
                    val[target],
                    model.predict(val.drop(target, axis=1)),
                    init_conf['tasktype'],
                    verbose=0)
                test_metrics = copy.deepcopy(metrics)
        else:
            if init_conf['model'] == 'prophet':
                train_metrics = copy.deepcopy(metrics)
                val_metrics = evaluate(
                    val[target],
                    model.predict(val.drop(target, axis=1), type = 'val'),
                    init_conf['tasktype'],  
                    verbose=0)
                test_metrics = None
            elif init_conf['model'] == 'xgboost':
                train_metrics = copy.deepcopy(metrics)
                val_metrics = evaluate(
                    val[target],
                    model.predict(val.drop(target, axis=1)),
                    init_conf['tasktype'],
                    verbose=0)
                test_metrics = None


        if init_conf['model'] == 'prophet':
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
        elif init_conf['model'] == 'xgboost':
            plot_regression_result(
                ytrainrescaled,
                model.predict(train.drop(target, axis=1)),
                yvalrescaled,
                model.predict(val.drop(target, axis=1)),
                train_metrics,
                val_metrics,
                test_metrics,
                ytestrescaled,
                model.predict(test.drop(target, axis=1)),
                modelname = init_conf['model'],
                savepath = str(init_conf['model'])+ '_'+ target + '_regression.png'  
            )
    

if __name__ == '__main__':
    main()
        
