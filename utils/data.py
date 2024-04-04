import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

from utils.data_processor_ineeji.raw_data_process import RawDataProcessor
from utils.data_processor_ineeji.boruta import Boruta
from utils.data_processor_ineeji.test_config import (
    prepro_conf,
    boruta_conf,
    model_conf
)

import torch

def load_data(file_path):
    return pd.read_csv(file_path)

def dataloader_to_dataframe(dataloader):
    """
    Convert dataloader to dataframe
    """
    features = []
    targets = []
    for featurestorch, targetstorch in dataloader:
        features.append(
            featurestorch.reshape(featurestorch.size(0), -1)
        ) # Flatten the features
        targets.append(targetstorch)
    # Concatenate lists of tensors into a single tensor
    X = torch.cat(features, dim=0)
    y = torch.cat(targets, dim=0)
    #converting to dataframe
    X = pd.DataFrame(X.numpy())
    y = pd.DataFrame(y.numpy())

    y.columns = ['target']
    data = pd.concat([X, y], axis = 1)
    return data

def preprocessing(
        rawdata, 
        target, 
        tasktype, 
        dropped_col = [], 
        train_val_test_ratio = (0.7, 0.1),
        fw_steps = 0,
        n_hist = 1,
        shuffle = False,
        apply_scaler = True, 
        returnasdf = False,
        date_col = None,
        preprocessed = False):
    """
    Future work: 
    - Add more preprocessing steps 
    - Add gating for if the data is already split into train and test
    - Add option to return as dataframe format from the RawDataProcessor class (done)
    """
    #before begin, drop columns that are not needed
    rawdata = rawdata.drop(dropped_col, axis = 1)
    
    if preprocessed == False:
        #take config from Cloud AI code
        base_conf = {}
        dataset_conf = {}
        
        prepro_conf["train_val_test_ratio"] = train_val_test_ratio
        prepro_conf["fw_steps"] = fw_steps
        prepro_conf["n_hist"] = n_hist
        prepro_conf["shuffle"] = shuffle
        prepro_conf["apply_scaler"] = apply_scaler
        prepro_conf["is_classification"] = bool(tasktype)
        if returnasdf:
            prepro_conf["change3d"] = False

        prepro_conf["target_column"] = target
        prepro_conf["target_column"] = (
        prepro_conf["target_column"]
        .replace(" ", "_")
        .replace("\n", "_")
        .replace("'", "_")
        .replace("__", "_")
        )

        prepro_conf['full_train_data_number'] = len(rawdata)

        base_conf['current_time'] = datetime.now().strftime("%Y_%m_%d_%H_%M")
        base_conf['dataset_name'] = target

        #feature selection
        boruta = Boruta(boruta_conf, prepro_conf)

        #fmt: off
        rawdataprocessor = RawDataProcessor(
        base_conf=base_conf,
        prepro_conf=prepro_conf,
        dataset_conf=dataset_conf,
        model_conf=model_conf,
        boruta=boruta,
        )

        # fmt: on
        if date_col is not None:
            return_date = True
        else:
            return_date = False
        
        (
            train_loader,
            val_loader,
            test_loader,
            _,
            _,
            _,
            scaler_y,
            _,
            train_date,
            val_date,
            test_date
        ) = rawdataprocessor.run_preprocessing(rawdata, 
                                            datename= date_col, 
                                            return_df = returnasdf,
                                            return_date = return_date)
        if tasktype == 1:
            scaler_y = None
        return train_loader, val_loader, test_loader, scaler_y, train_date, val_date, test_date
    else:
        train_index = int(len(rawdata) * train_val_test_ratio[0])
        val_index = train_index + int(len(rawdata) * train_val_test_ratio[1])

        train = rawdata.iloc[:train_index]
        if shuffle:
            train = train.sample(frac=1).reset_index(drop=True)

        val = rawdata.iloc[train_index:val_index]

        test = rawdata.iloc[val_index:]
        
        if date_col is not None:
            train[date_col] = pd.to_datetime(train[date_col])
            train_date = train[date_col]

            val_date = val[date_col]
            val[date_col] = pd.to_datetime(val[date_col])

            test_date = test[date_col]
            test[date_col] = pd.to_datetime(test[date_col])
        else:
            train_date = None
            val_date = None
            test_date = None
        
        return train, val, test, None, train_date, val_date, test_date