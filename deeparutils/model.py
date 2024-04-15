import numpy as np
import pandas as pd
import pickle
import warnings

import torch 
import lightning.pytorch as pl

from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_lightning import loggers as pl_loggers

from hyperopt import hp, fmin, tpe, Trials, space_eval
from sklearn.metrics import mean_squared_error

from utils.utilitycode import convert_to_hp_space
from deeparutils.time_and_dummy import generate_time_group_idx

class DeepARModel():
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
        self.scalertarget = scalertarget
        if cuda:
            self.device = 'gpu'
        else:
            self.device = 'cpu'

        self.train_date = train_date
        self.val_date = val_date   
        self.test_date = test_date

        warnings.filterwarnings('ignore')

    def reformat_data(self,
                      traindata,
                      valdata,
                      testdata,
                      target): #reformatting data to fit to the package requirements
        #add time_idx and dummy group id
        #for now, the time_idx is incremental by row and group_id is constant
        #find out a better way later
        self.target = target
        self.traindata, self.valdata, self.testdata = (
            generate_time_group_idx(traindata,
                                    valdata,
                                    self.train_date,
                                    self.val_date,
                                    test = testdata,
                                    test_date = self.test_date)
        )

        #adjusting evaluation column
        #the nhist will have effect on how the model predicts
        #adjust valdata and test data later for evaluation based on number of nhist.
        self.valdata = pd.concat([self.traindata.iloc[-self.nhist:], self.valdata], axis = 0, ignore_index=True)
        if testdata is not None:
            self.testdata = pd.concat([self.valdata.iloc[-self.nhist:], self.testdata], axis = 0, ignore_index=True)

        #create dataset
        self.training = self.make_dataset(self.traindata, self.target)
        self.validation = self.make_dataset(self.valdata, self.target)
        if testdata is not None:
            self.testing = self.make_dataset(self.testdata, self.target)

        #create dataloader
        self.train_loader = self.training.to_dataloader(train = True, 
                                                        batch_size = self.batch_size, 
                                                        num_workers = self.num_workers)
        self.val_loader = self.validation.to_dataloader(train = False, 
                                                        batch_size = self.batch_size, 
                                                        num_workers = self.num_workers)
        if testdata is not None:
            self.test_loader = self.testing.to_dataloader(train = False, 
                                                          batch_size = self.batch_size, 
                                                          num_workers = self.num_workers)

    def hyperparameter_tuning(self, hyperparameters):
        #first delete lightning_logs if exist
        self.cleardir()

        hyperparameters = convert_to_hp_space(hyperparameters)
        pl.seed_everything(42)

        def objective(params, self = self): #later add more flexibility to hyperparameter tuning
            model = DeepAR.from_dataset(
                self.training, rnn_layers=params['rnn_layers'], hidden_size=int(params['hidden_size']),
                dropout=params['dropout'], learning_rate=params['learning_rate'], weight_decay=params['weight_decay'],
                log_interval = 10, log_val_interval = 1, loss = NormalDistributionLoss())
            
            #create trainer
            tensorboard = pl_loggers.TensorBoardLogger('lightning_logs')
            trainer = pl.Trainer(
                max_epochs = self.epochs,
                accelerator = self.device,
                enable_model_summary = False,
                logger= tensorboard,
            )
            #train the model
            trainer.fit(model, self.train_loader)
            #predict
            actuals = self.valdata[self.target].iloc[self.nhist:]

            predictions = model.predict(self.val_loader, mode = 'prediction')
            predictionstoseries = pd.Series(predictions.cpu().numpy().flatten())

            actuals = self.scalertarget.inverse_transform(actuals.values.reshape(-1, 1)).flatten()
            predictions = self.scalertarget.inverse_transform(predictionstoseries.values.reshape(-1,1)).flatten()

            #calculate rmse
            rmse = mean_squared_error(actuals, predictions, squared = False)
            return {'loss': rmse, 'status': 'ok'}
        trials = Trials()
        best_params = fmin(fn=objective, space = hyperparameters, algo = tpe.suggest, 
                           max_evals = 3, trials=trials, rstate = np.random.default_rng(42))
        print('Hyperparameter tuning completed!')
        self.params = space_eval(hyperparameters, best_params)

    def train(self):
        self.cleardir()
        self.model = DeepAR.from_dataset(
            self.training, rnn_layers=self.params['rnn_layers'], hidden_size=int(self.params['hidden_size']),
            dropout=self.params['dropout'], learning_rate=self.params['learning_rate'], weight_decay=self.params['weight_decay'],
            log_interval = 10, log_val_interval = 1, loss = NormalDistributionLoss()
            )
        
        tensorboard = pl_loggers.TensorBoardLogger('lightning_logs')
        trainer = pl.Trainer(
            max_epochs = self.epochs,
            accelerator = self.device,
            enable_model_summary = True,
            logger= tensorboard,
        )
        trainer.fit(self.model, self.train_loader)
    
    def predict(self, type = 'test'):
        print(f'Predicting on {type} data...')
        self.model.eval()
        if type.lower() == 'train':
            predictions = self.model.predict(self.train_loader, mode = 'prediction')
        if type.lower() == 'val':
            predictions = self.model.predict(self.val_loader, mode = 'prediction')
        if type.lower() == 'test':
            predictions = self.model.predict(self.test_loader, mode = 'prediction')
        predictionstoseries = pd.Series(predictions.cpu().numpy().flatten())
        predictions = self.scalertarget.inverse_transform(predictionstoseries.values.reshape(-1,1)).flatten()
        return pd.Series(predictions)

    def set_additional_param(self, nhist, fw_steps, batch_size, num_workers, epochs):
        self.nhist = nhist
        self.fw_steps = fw_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
    
    def make_dataset(self, data, target):
        dataset = TimeSeriesDataSet(
            data,
            time_idx = 'time_idx',
            target = target,
            group_ids = ['group_id'],
            min_encoder_length = self.nhist,
            max_encoder_length = self.nhist,
            min_prediction_length = self.fw_steps,
            max_prediction_length = self.fw_steps,
            time_varying_known_reals = data.columns.difference([target, 'group_id']).tolist(),
            time_varying_unknown_reals = [target]
            )
        return dataset
    
    def cleardir(self):
        import os
        import shutil
        if os.path.exists('lightning_logs'):
            shutil.rmtree('lightning_logs',ignore_errors=True)

    def save_model(self):
        print('Save model is not yet implemented for saving the model')
