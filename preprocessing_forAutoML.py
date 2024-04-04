from utils.data import load_data, preprocessing
from utils.evaluate import evaluate

import argparse
import yaml
import pandas as pd
from joblib import dump

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
def main():
    ###check chosen model
    if init_conf['tasktype'].lower() == 'classification':
        init_conf['tasktype'] = 1 #classification
    elif init_conf['tasktype'].lower() == 'regression':
        init_conf['tasktype'] = 0 #regression
    else:
        raise ValueError('Task type not recognized. Please choose either classification or regression')
    # load data
    rawtrain = load_data(data_conf['traindir'])
    if data_conf['testdir'] is not None:
        rawtest = load_data(data_conf['testdir'])

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
                      date_col=data_conf['date_col']                      
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


    print("Combine train, val, test!") 
    df = pd.concat([train, val, test], ignore_index=True, axis=0)
    #add the date if train_date, val_date, test_date is not None
    if train_date is not None and val_date is not None and test_date is not None:
        dfdate = pd.concat([train_date, val_date, test_date], ignore_index=True, axis=0)
        print('printing dfdate:', dfdate)
        df['date'] = dfdate
    print("Final combined data shape: ", df.shape)

    df.to_csv(data_conf['traindir'].split('.')[0]+'_preprocessed.csv', index=False)
    #save the scaler for y
    dump(scaler_y, data_conf['traindir'].split('.')[0]+'_scaler_y.joblib')


if __name__ == '__main__':
    main()
