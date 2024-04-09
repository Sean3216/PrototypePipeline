import pandas as pd

#add a time_idx
def generate_time_group_idx(train, val, train_date, val_date, test = None, test_date = None):
    ### This function is still questionable since it sets the time_idx to be incremental by row
    #grabbing the index for resplitting
    trainlastindex = len(train)-1
    vallastindex = len(val)+trainlastindex

    if test is None:
        df = pd.concat([train, val], axis=0, ignore_index=True)
        if train_date is not None and val_date is not None:
            time_date = pd.concat([train_date, val_date], axis=0, ignore_index=True)
    else:
        df = pd.concat([train, val, test], axis=0, ignore_index=True)
        if train_date is not None and val_date is not None and test_date is not None:
            time_date = pd.concat([train_date, val_date, test_date], axis=0, ignore_index=True)

    df.reset_index(drop=True, inplace=True)
    if train_date is not None and val_date is not None:
        time_date.reset_index(drop=True, inplace=True)
        df['datecol'] = time_date
        df.sort_values('datecol', inplace=True)

    #for increment time_idx by date
    df['time_idx'] = range(0, len(df))

    if 'datecol' in df.columns:
        #drop datecol
        df = df.drop(columns=['datecol'])
    #add dummy group id
    df = add_dummy_group(df)

    train = df.loc[:trainlastindex]

    if test is None:
        val = df.loc[trainlastindex+1:]
        return train, val
    else:
        val = df.loc[trainlastindex+1:vallastindex]
        test = df.loc[vallastindex+1:]
    return train, val, test

def add_dummy_group(df):
    df['group_id'] = 1
    return df