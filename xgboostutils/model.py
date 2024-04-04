import pandas as pd
import numpy as np
import cupy as cp
import xgboost as xgb
from hyperopt import fmin, tpe, Trials, space_eval
from sklearn.metrics import accuracy_score, root_mean_squared_error
from utils.utilitycode import convert_to_hp_space

class XGBoostModel():
    #@Todo: Add hyperparameter tuning capability
    def __init__(self,
                 tasktype,
                 params = None, 
                 model_path = None,
                 load_model = False,
                 cuda = False,
                 scalertarget = None):
        self.params = params
        self.tasktype = tasktype
        self.model_path = model_path
        self.load_model = load_model    
        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.scalertarget = scalertarget

    def train(self, X_train, y_train):
        if self.tasktype == 1:
            if self.params is None:
                self.model = xgb.XGBClassifier(device = self.device)
            else:
                self.model = xgb.XGBClassifier(**self.params, device = self.device)
        else:
            if self.params is None:
                self.model = xgb.XGBRegressor(device = self.device)
            else:
                self.model = xgb.XGBRegressor(**self.params, device = self.device)

        if self.load_model:
            print('Asked to load model. Loading model from path...')
            self.model.load_model(self.model_path) #load model if model path is provided
        self.model.fit(X_train, y_train)
        print('Model trained successfully!')
    
    def predict(self, X_test, threshold = None): #in case we want to tweak the threshold of the classification model
        if self.device == 'cuda':
            X_test = cp.array(X_test)
        if threshold is not None:
            if self.tasktype == 1:
                y_pred = self.model.predict_proba(X_test)
                y_pred = (y_pred[:,1] >= threshold).astype('int')
            elif self.tasktype == 0:
                print('Threshold is only applicable for classification task.')
                print('Predicting without threshold...')    
                y_pred = self.model.predict(X_test)
        else:
            print('No threshold provided.')
            print('Predicting without threshold...')
            y_pred = self.model.predict(X_test)
            if self.tasktype == 0:
                if self.scalertarget is not None:
                    #fed a 2-dimensional array to inverse_transform. y_pred is a series
                    y_pred = self.scalertarget.inverse_transform(y_pred.reshape(-1, 1))
                    y_pred = pd.Series(y_pred.flatten())
        return y_pred 
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, hyperparameters):  
        hyperparameters = convert_to_hp_space(hyperparameters)  
        if self.device == 'cuda':
            X_val = cp.array(X_val)
        if self.scalertarget is not None:
            if self.tasktype == 0:
                y_val = self.scalertarget.inverse_transform(y_val.values.reshape(-1, 1))
                y_val = pd.Series(y_val.flatten())
        def objective(params):
            if self.tasktype == 1:
                self.model = xgb.XGBClassifier(**params, device = self.device)
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                score = -1 * accuracy_score(y_val, y_pred)
            else:
                self.model = xgb.XGBRegressor(**params, device = self.device)
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                if self.scalertarget is not None:
                    ypredrescaled = self.scalertarget.inverse_transform(y_pred.reshape(-1, 1))
                    ypredrescaled = pd.Series(ypredrescaled.flatten())
                else:
                    ypredrescaled = y_pred
                score = root_mean_squared_error(y_val, ypredrescaled)
            return {'loss': score, 'status': 'ok'}

        trials = Trials()
        best_params = fmin(
            objective, 
            hyperparameters, 
            algo=tpe.suggest, 
            max_evals=100, 
            trials=trials, 
            rstate = np.random.default_rng(42)
        )
        print('Hyperparameter tuning completed!')
        converted_params = space_eval(hyperparameters, best_params)
        self.params = converted_params
        print('Self.params updated? ', self.params == converted_params)
        print('Best hyperparameters: {}'.format(converted_params))

    def save_model(self):
        if self.model_path is not None:
            self.model.save_model(self.model_path)
            print('Model saved successfully!')
        else:
            print('Model path not provided. Model not saved.')