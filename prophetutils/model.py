import pandas as pd
import numpy as np
import logging
import warnings
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from hyperopt import fmin, tpe, Trials, space_eval
from sklearn.metrics import accuracy_score, mean_squared_error
from utils.utilitycode import convert_to_hp_space

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').disabled = True
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class ProphetModel():
    def __init__(self,
                 tasktype,
                 params = None, 
                 model_path = None,
                 load_model = False,
                 cuda = False,
                 scalertarget = None,
                 train_date = None,
                 val_date = None,
                 test_date = None):
        self.params = params
        self.tasktype = tasktype
        self.model_path = model_path
        self.load_model = load_model    
        if cuda:
            print('CUDA not yet available for Prophet. Using CPU...')
        self.scalertarget = scalertarget
        
        self.train_date = train_date
        self.val_date = val_date
        self.test_date = test_date


    def train(self, X_train, y_train):
        if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            ValueError('X_train must be a pandas dataframe')
        #rename y_train to y
        y_train = y_train.rename('y')
        #use train date for prophet
        if self.train_date is None:
            #create a dummy date column
            self.train_date = pd.date_range(start = '1/1/1945', periods = len(X_train), freq = 'D')
        X_train['ds'] = self.train_date
        df = pd.concat([X_train, y_train], axis = 1)
        #convert all the columns name to string
        df.columns = [str(i) for i in df.columns]
        if self.load_model:
            print('Asked to load model. Loading model from path...')
            with open(self.model_path, 'rb') as file:
                self.model = model_from_json(file.read())
        else:
            if self.params is None:
                self.model = Prophet()
            else:  
                self.model = Prophet(**self.params)
        #check if there are any extra regressors
        if len(df.drop(['ds', 'y'], axis = 1).columns) > 0:
            extra_regressors = df.drop(['ds', 'y'], axis = 1).columns.tolist()
            for regressor in extra_regressors:
                self.model.add_regressor(regressor)
        self.model.fit(df)
        print('Model trained successfully!')
    
    def predict(self, X_test, type = 'test'): #in case we want to tweak the threshold of the classification model
        print(f'Predicting on {type} data...')
        #use test date for prophet
        if type.lower() == 'train':
            X_test['ds'] = self.train_date
            df = X_test.copy()
        elif type.lower() == 'test':
            if self.test_date is None:
                #create a dummy date column. Start date is the last date of val date
                self.test_date = pd.date_range(start = self.val_date[-1], periods = len(X_test), freq = 'D')
            X_test['ds'] = self.test_date
            df = X_test.copy()
        elif type.lower() == 'val':
            X_test['ds'] = self.val_date
            df = X_test.copy()
        else:
            ValueError('Type of data must be either train, val or test')
        df.columns = [str(i) for i in df.columns]
        y_pred = self.model.predict(df)
        if self.scalertarget is not None:
            y_predscaled = self.scalertarget.inverse_transform(y_pred['yhat'].values.reshape(-1, 1))
            y_predscaled = pd.Series(y_predscaled.flatten())
        else:
            y_predscaled = y_pred['yhat']
        return y_predscaled
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, hyperparameters):  
        hyperparameters = convert_to_hp_space(hyperparameters)  
        if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
            ValueError('X_train must be a pandas dataframe')
        #rename y_train to y
        y_train = y_train.rename('y')
        #use train date for prophet
        if self.train_date is None:
            #create a dummy date column
            self.train_date = pd.date_range(start = '1/1/1945', periods = len(X_train), freq = 'D')
        X_train['ds'] = self.train_date
        print("Train date: ", X_train['ds'])
        print('Number of missing in date: ', X_train['ds'].isnull().sum())
        df_train = pd.concat([X_train, y_train], axis = 1)
        #convert all the columns name to string
        df_train.columns = [str(i) for i in df_train.columns]

        #use val date for prophet
        if self.val_date is None:
            #create a dummy date column. Start date is the last date of train date
            self.val_date = pd.date_range(start = self.train_date[-1], periods = len(X_val), freq = 'D')
        X_val['ds'] = self.val_date
        print("Val date: ", X_val['ds'])
        print('Number of missing in date: ', X_val['ds'].isnull().sum())
        df_val = X_val.copy()
        df_val.columns = [str(i) for i in df_val.columns]
        def objective(params):
            self.model = self.model = Prophet(**params)
            if df_train.columns.difference(['ds', 'y']).any():
                extra_regressors = df_train.columns.difference(['ds', 'y']).tolist()
                extra_regressors = [str(i) for i in extra_regressors]
                for regressor in extra_regressors:
                    self.model.add_regressor(regressor)
            self.model.fit(df_train)
            y_pred = self.model.predict(df_val)
            if self.scalertarget is not None:
                y_predscaled = self.scalertarget.inverse_transform(y_pred['yhat'].values.reshape(-1, 1))
                y_predscaled = pd.Series(y_predscaled.flatten()).astype('float')
                y_valscaled = self.scalertarget.inverse_transform(y_val.values.reshape(-1, 1))
                y_valscaled = pd.Series(y_valscaled.flatten()).astype('float')
            else:
                y_predscaled = y_pred['yhat']
                y_valscaled = y_val
            score = np.sqrt(mean_squared_error(y_valscaled, y_predscaled))
            return {'loss': score, 'status': 'ok'}

        trials = Trials()
        best_params = fmin(
            objective, 
            hyperparameters, 
            algo=tpe.suggest, 
            max_evals=10, 
            trials=trials, 
            rstate = np.random.default_rng(42)
        )
        print('Hyperparameter tuning completed!')
        converted_params = space_eval(hyperparameters, best_params)
        print('self.params before update: ', self.params)
        self.params = converted_params
        print('self.params after update: ', self.params)
        print('Self.params updated? ', self.params == converted_params)
        print('Best hyperparameters: {}'.format(converted_params))

    def save_model(self):
        if self.model_path is not None:
            print('Code not yet implemented. Model not saved.')
        #    with open(str(self.model_path), 'wb') as file:
        #        file.write(model_to_json(self.model))
        else:
            print('Model path not provided. Model not saved.')
