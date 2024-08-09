import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


class RawDataProcessor:
    def __init__(
        self,
        base_conf,
        prepro_conf,
        dataset_conf,
        model_conf,
        boruta,
    ):
        self.base_conf = base_conf
        self.prepro_conf = prepro_conf
        self.dataset_conf = dataset_conf
        self.model_conf = model_conf
        self.boruta = boruta
        # self.bayesian = bayesian
        self.set_scaler()

    def set_scaler(
        self,
    ):
        def get_scaler(scaler_type, apply_scaler=True):
            scaler_dict = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }
            return deepcopy(scaler_dict[scaler_type]) if apply_scaler else None

        self.x_scaling = get_scaler(self.prepro_conf["scaler_name"])
        self.y_scaling = get_scaler(self.prepro_conf["scaler_name"])

    def run_preprocessing(self, df=None, return_df=False, datename = None, return_date = False, filename = 'sample_data.csv'): # modified by Yonathan (add return_df and return_date)
        if datename is None:
            checkdate = self.check_date_columns(df)
            if checkdate == []:
                print("No date column found in the dataset")
                datename = []
            else:
                datename = checkdate.copy()
        
        #sampling 14000 rows for faster processing
        if len(df) > 14000:
            df = df.sample(14000, random_state=42)
            print("Data is sampled for faster processing")
            if datename is not None or datename != []:
                #sort by date column
                df = df.sort_values(by=datename, ascending=True)
            else:
                #sort by index
                df = df.sort_index()
            df.reset_index(drop=True, inplace=True)
            df.to_csv('toshare/'+filename.split('/')[1]+'.csv',index = False)

        #this is only for AI expo
        self.filename = filename

        print(f"Original data shape: {df.shape}")

        if df is None:
            # 1. load_data
            df, dataset_name = self.load_data(self.base_conf, self.dataset_conf)
            self.base_conf["dataset_name"] = dataset_name
            # dataset_name='df'
            # print("EEEEEEE####################")
            # print(self.prepro_conf["target_column"])

            print(self.prepro_conf["dataset_key"])
            if "dataset_key" in self.prepro_conf:
                print("updateing prepro_conf with dataset_key")
                self.prepro_conf.update(
                    self.dataset_conf[self.prepro_conf["dataset_key"]]
                )
            # if ["is_classification"] doesnt exist in self.prepro_conf, then it assert to add prepro_conf
        assert (
            "is_classification" in self.prepro_conf
        ), "Please add 'is_classification' to prepro_conf"
        # 2. outlier 제거, reg or cls 분리, fw step 처리
        cleaned_df, target_column, num_classes, label_mapping = (
            self.prepare_data(  # num_calsses, label_mapping은 classification일때만 사용
                df, target_column=self.prepro_conf["target_column"], date_name = datename
            )
        )
        if self.prepro_conf["fw_steps"] > 0:
            print("Data is shifted for forward prediction")
            future_step = self.prepro_conf["fw_steps"]
            print(f"future_step: {future_step}")
            shifted_df = cleaned_df.copy()
            target_column_shifted = "shifted"
            shifted_df[target_column_shifted] = shifted_df[target_column].shift(
                -future_step
            )
            shifted_df.drop([target_column], axis=1, inplace=True)

            # Use dropna to remove rows with NaN values
            shifted_df = shifted_df.dropna()
            #create a copy of cleaned_df for boruta without datecol
            copy_df = shifted_df.copy()
            if type(datename) == list:
                for date in datename:
                    if date in copy_df.columns:
                        copy_df = copy_df.drop(date, axis=1)
            else:
                if datename in copy_df.columns:
                    copy_df = copy_df.drop(datename, axis=1)

            self.selected_features = self.boruta.run_boruta(
                copy_df,
                target_column_shifted,
            )

            # self.selected_features = self.selected_features.append(pd.Index([target_column_shifted]))

            if target_column_shifted in self.selected_features:
                self.selected_features = self.selected_features.drop(
                    target_column_shifted
                )

            shifted_df.rename(
                columns={target_column_shifted: target_column}, inplace=True
            )
            cleaned_df = shifted_df.copy()
            del shifted_df
        else:
            # 3. feature selection by boruta
            #create a copy of cleaned_df for boruta without datecol
            copy_df = cleaned_df.copy()
            if type(datename) == list:
                for date in datename:
                    if date in copy_df.columns:
                        copy_df = copy_df.drop(date, axis=1)
            else:
                if datename in copy_df.columns:
                    copy_df = copy_df.drop(datename, axis=1)
            
            self.selected_features = self.boruta.run_boruta(
                copy_df,
                target_column,
            )
            if target_column in self.selected_features:
                self.selected_features = self.selected_features.drop(target_column)
        # 4. bayesian optimization

        print(cleaned_df.columns)
        print(cleaned_df.head(3))

        # 5. data_loader or dataframe # modified by Yonathan (add condition if return_df is False)
        (
            train_loader,
            val_loader,
            test_loader,
            input_size,
            out,
            scaler_x,
            scaler_y,
            train_date,
            val_date,
            test_date
        ) = self.data_loader(
            cleaned_df,
            self.selected_features,  # 실제 기존 target
            target_column,  # windowing, fw step 처리한 target
            num_classes,
            isreturndf = return_df,
            isreturndate = return_date,
            datecolname = datename
        )
        self.prepro_conf["number_of_classes"] = out

        return (
            train_loader,
            val_loader,
            test_loader,
            input_size,
            out,
            scaler_x,
            scaler_y,
            label_mapping,
            train_date,
            val_date,
            test_date
        )

    def load_data(self, base_conf, dataset_conf):
        #  = os.path.splitext(file_path)
        dataset_key = self.prepro_conf["dataset_key"]
        assert (
            dataset_key in dataset_conf
        ), f"dataset_key {dataset_key} is not in dataset_conf"
        file_path = f"data/{dataset_key}.csv"
        # dataset_conf에 있는 key, value pair모두를 가져와서 prepro_conf에 update
        self.prepro_conf["file_path"] = file_path

        dataset_name, file_extension = os.path.splitext(os.path.basename(file_path))
        dir_name = os.path.dirname(file_path)
        exit_files = os.listdir(dir_name)
        if dataset_name + ".parquet" in exit_files:
            print("The file is already in Parquet format.")
            df = pd.read_parquet(file_path.replace(file_extension, ".parquet"))
            return df, dataset_name
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        # elif file_extension == ".parquet":
        #     print("The file is already in Parquet format.")
        #     df = pd.read_parquet(file_path)
        #     return df, dataset_name
        else:
            raise Exception("Unsupported file format.")

        # Save the DataFrame as a Parquet file
        save_path_as_par = os.path.splitext(file_path)[0] + ".parquet"
        df.to_parquet(save_path_as_par, index=False)
        print(f"File saved as {save_path_as_par}")

        return df, dataset_name

    def prepare_data(self, df, target_column, date_name = []):
        """
        Prepares the data based on whether it is for a classification problem or not.
        """
        # Assuming `process_data_classification` and `checkfeatures` are predefined utils for handling classification data
        # Calculate the threshold for dropping columns (70% of total entries)
        threshold = len(df) * 0.7
        # Drop columns with missing values exceeding the threshold
        df = df.dropna(thresh=threshold, axis=1)

        if self.prepro_conf["is_classification"]:
            df = df.dropna(axis=1)
        else:
            # Select only numerical columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            # Temporarily convert integers to floats for interpolation
            df[numerical_columns] = df[numerical_columns].astype(float)
            # Apply interpolation to fill NaNs with the mean of values before and after the NaN
            df[numerical_columns] = df[numerical_columns].apply(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )

        #df.reset_index(drop=True, inplace=True) #try to not reset index
        df.columns = (
            df.columns.str.replace(" |\n", "_", regex=True)
            .str.replace("'", "_")
            .str.replace("__", "_")
            .str.replace(".", "_")
        )
        if self.prepro_conf["is_classification"]:
            cleaned_df, label_mapping, _, num_classes = (
                self.process_data_classification(df, target_column)
            )
            #cleaning column names again if there are any new columns
            target = cleaned_df[target_column]
            cleaned_df = cleaned_df.drop(target_column, axis=1)
            cleaned_df.columns = (
                cleaned_df.columns.str.replace(" |\n", "_", regex=True)
                .str.replace("'", "_")
                .str.replace("__", "_")
                .str.replace(".", "_")
            )
            cleaned_df[target_column] = target
            return cleaned_df, target_column, num_classes, label_mapping
        else:
            cleaned_df = self.remove_outliers_iqr(df, target_column)
            cleaned_df = cleaned_df.dropna()
            if date_name is not None or date_name != []:
                if type(date_name) == list:
                    for date in date_name:
                        cleaned_df[date] = pd.to_datetime(cleaned_df[date])
                else:
                    cleaned_df[date_name] = pd.to_datetime(cleaned_df[date_name])
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == "object":
                    dummies = pd.get_dummies(cleaned_df[col], prefix=col)
                    cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
                    cleaned_df = cleaned_df.drop(col, axis=1)
            #cleaning column names again if there are any new columns
            target = cleaned_df[target_column]
            cleaned_df = cleaned_df.drop(target_column, axis=1)
            cleaned_df.columns = (
                cleaned_df.columns.str.replace(" |\n", "_", regex=True)
                .str.replace("'", "_")
                .str.replace("__", "_")
                .str.replace(".", "")
            )
            cleaned_df[target_column] = target
            return cleaned_df, target_column, None, None

    def data_loader(self, cleaned_df, selected_features, target_column, num_classes, isreturndf = False, isreturndate = False, datecolname = []):
        n_hist, fw_steps, if_multiple = validate_data_parameters(
            self.prepro_conf["n_hist"],
            self.prepro_conf["fw_steps"],
            self.prepro_conf["if_multiple"],
        )
        (
            self.prepro_conf["n_hist"],
            self.prepro_conf["fw_steps"],
            self.prepro_conf["if_multiple"],
        ) = (n_hist, fw_steps, if_multiple)

        print("Data parameters validated")
        # print("n_hist:", n_hist, "fw_steps:", fw_steps, "if_multiple:", if_multiple)
        scaled_data, datedict = self.preprocess_data(
            cleaned_df,
            full_train_data_number=self.prepro_conf["full_train_data_number"],
            selected_features=selected_features,
            target_column=target_column,
            num_classes=num_classes,
            change3d = self.prepro_conf['change3d'], # modified by Yonathan (add change3d),
            returningdate = isreturndate,
            date_name = datecolname
        )
        print("Data preprocessed")
        print(scaled_data["s_y_train"][:20])

        if self.prepro_conf["combine_val_train"]:
            new_val_x = np.concatenate(
                (scaled_data["s_x_train"], scaled_data["s_x_val"])
            )
            new_val_y = np.concatenate(
                (scaled_data["s_y_train"], scaled_data["s_y_val"])
            )

        if isreturndf: # modified by Yonathan (add condition if isreturndf is True)
            if self.prepro_conf['change3d']:
                raise ValueError("returning dataframe is not supported for 3d data")
            train_loader = pd.DataFrame(scaled_data["s_x_train"], columns=selected_features)
            train_loader[target_column] = scaled_data["s_y_train"]
            
            val_loader = pd.DataFrame(scaled_data["s_x_val"], columns=selected_features)
            val_loader[target_column] = scaled_data["s_y_val"]
  
            test_loader = pd.DataFrame(scaled_data["s_x_test"], columns=selected_features)
            test_loader[target_column] = scaled_data["s_y_test"]
        else:
            if self.prepro_conf["is_classification"]:
                train_loader = dataloader_classification(
                    scaled_data["s_x_train"],
                    scaled_data["s_y_train"],
                    self.model_conf["base_batch_size"],
                    shuffle=self.prepro_conf["shuffle"],
                )
                val_loader = dataloader_classification(
                    scaled_data["s_x_val"],
                    scaled_data["s_y_val"],
                    self.model_conf["base_batch_size"],
                    shuffle=False,
                )
                test_loader = dataloader_classification(
                    scaled_data["s_x_test"],
                    scaled_data["s_y_test"],
                    self.model_conf["base_batch_size"],
                    shuffle=False,
                )
            else:
                train_loader = dataloader_regression(
                    scaled_data["s_x_train"],
                    scaled_data["s_y_train"],
                    self.model_conf["base_batch_size"],
                    shuffle=self.prepro_conf["shuffle"],
                )
                if self.prepro_conf["combine_val_train"]:
                    val_loader = dataloader_regression(
                        new_val_x,
                        new_val_y,
                        self.model_conf["base_batch_size"],
                        shuffle=False,
                    )
                else:
                    val_loader = dataloader_regression(
                        scaled_data["s_x_val"],
                        scaled_data["s_y_val"],
                        self.model_conf["base_batch_size"],
                        shuffle=False,
                    )
                test_loader = dataloader_regression(
                    scaled_data["s_x_test"],
                    scaled_data["s_y_test"],
                    self.model_conf["base_batch_size"],
                    shuffle=False,
                )
        return (
            train_loader,
            val_loader,
            test_loader,
            scaled_data["input_size"],
            scaled_data["out"],
            scaled_data["scaler_x"],
            scaled_data["scaler_y"],
            datedict['train_date'],
            datedict['val_date'],
            datedict['test_date']
        )

    def preprocess_data(
        self,
        cleaned_df,
        full_train_data_number,
        selected_features,
        target_column,
        num_classes,
        change3d = True,
        returningdate = False,
        date_name = [],
    ):

        if full_train_data_number > len(cleaned_df):
            full_train_data_number = len(cleaned_df)
        #this is only for AI expo. Delete later
        cleaned_df = cleaned_df.reset_index()
        #################################################
        cleaned_df_subset = cleaned_df.iloc[:full_train_data_number]
        #################################################


        # Xs_shape, ys_shape = calculate_3d_shape(
        #     full_train_data_number,
        #     self.prepro_conf["n_hist"],
        #     self.prepro_conf["fw_steps"],
        #     self.prepro_conf["if_multiple"],
        # )
        # Calculate indices for train, val, test split
        total_len = full_train_data_number
        train_ratio, valid_ratio = self.prepro_conf["train_val_test_ratio"]
        train_end_idx = int(total_len * train_ratio)
        valid_end_idx = int(total_len * (train_ratio + valid_ratio))

        #this is also for AI expo. Delete later
        #print the train, val, test index
        dummy = cleaned_df_subset.copy()
        cleaned_df_subset.drop('index', axis=1, inplace=True)

        traindummy = dummy.iloc[:train_end_idx]
        valdummy = dummy.iloc[train_end_idx:valid_end_idx]
        testdummy = dummy.iloc[valid_end_idx:]

        traindummy = traindummy.set_index('index')
        valdummy = valdummy.set_index('index')
        testdummy = testdummy.set_index('index')
        print("Train index:", traindummy.index.tolist())
        print("Val index:", valdummy.index.tolist())
        print("Test index:", testdummy.index.tolist())
        #export to pickle
        traindummy.to_pickle('toshare/'+self.filename.split('/')[1]+'_train_index.pkl')
        valdummy.to_pickle('toshare/'+self.filename.split('/')[1]+'_val_index.pkl')
        testdummy.to_pickle('toshare/'+self.filename.split('/')[1]+'_test_index.pkl')

        # Select features and target
        if returningdate:
            if date_name != []:
                print(f'Returning date series as per configuration. Date column name: {date_name}')
                dateseries = cleaned_df_subset[date_name]
                cleaned_df_subset = cleaned_df_subset.drop(date_name, axis=1)
            else:
                print("Date column name is not provided and cannot be found!")
                print("Date cannot be returned.")
                dateseries = None
        else:
            print('Value of date_name:', date_name)
            if date_name != []:
                if type(date_name) == list and len(date_name) > 1:
                    print("Date series is not return because multiple date columns are found.") #later fix this. This is a quick fix
                    dateseries = None
                else:
                    print("Date columns exist but was not provided in the configuration file.")
                    userfeedback = input("Do you want to return date series? (y/n): ")
                    if userfeedback.lower() == 'y':
                        print("Returning date series because user asks it.")
                        if date_name[0] in cleaned_df_subset.columns: #quick fix
                            dateseries = cleaned_df_subset[date_name[0]]
                            cleaned_df_subset = cleaned_df_subset.drop(date_name[0], axis=1)
                        else:
                            print("Date is not returned because date column name is not found.") #later fix this. This is a quick fix
                            dateseries = None
                    else:
                        print("Date is not returned.")
                        dateseries = None
                        for date in date_name:
                            if date in cleaned_df_subset.columns:
                                cleaned_df_subset = cleaned_df_subset.drop(date, axis=1)
            else:
                print("Date is not returned because date column name is not provided and cannot be found.")
                dateseries = None

        X = cleaned_df_subset[selected_features].to_numpy()
        y = cleaned_df_subset[target_column].to_numpy()

        if self.prepro_conf["apply_scaler"]:
            print(f"scaling applied")
            self.x_scaling.fit(X[:train_end_idx])
            self.y_scaling.fit(y[:train_end_idx].reshape(-1, 1))  # y는 무조건 1 feature
            X_scaled = self.x_scaling.transform(X)
            y_scaled = self.y_scaling.transform(
                y.reshape(-1, 1)
            ).flatten()  # Flatten y back to original shape after scaling

        else:
            print(f"scaling Non Applied")
            X_scaled = X
            y_scaled = y.reshape(-1, 1).flatten()

        if self.prepro_conf["is_classification"]:
            y_scaled = y.astype(int)
        print(
            f"preprocess: backsteps: {self.prepro_conf['n_hist']}, fw_steps: {self.prepro_conf['fw_steps']}, if_multiple: {self.prepro_conf['if_multiple']}"
        )
        if change3d:
            X_3d, y_3d = create_3d_data(
                X_scaled,
                y_scaled,
                is_classification=self.prepro_conf["is_classification"],
                back_steps=self.prepro_conf["n_hist"],
                fw_steps=self.prepro_conf["fw_steps"],
            )
        else:
            if self.prepro_conf['n_hist'] > 1 or self.prepro_conf['fw_steps'] > 0:
                print(f'n_hist is {self.prepro_conf["n_hist"]} and fw_steps is {self.prepro_conf["fw_steps"]} but change3d is False.')
                print('Returning the data as 2D without backward or forward steps.')
            X_3d, y_3d = X_scaled, y_scaled

        # Split the 3D data
        x_train_3d, y_train_3d = X_3d[:train_end_idx], y_3d[:train_end_idx]

        x_val_3d, y_val_3d = (
            X_3d[train_end_idx:valid_end_idx],
            y_3d[train_end_idx:valid_end_idx],
        )
        x_test_3d, y_test_3d = X_3d[valid_end_idx:], y_3d[valid_end_idx:]
        
        #Split date series
        if dateseries is not None:
            train_date = dateseries[:train_end_idx]
            val_date = dateseries[train_end_idx:valid_end_idx]
            test_date = dateseries[valid_end_idx:]
            date={
                "train_date": train_date,
                "val_date": val_date,
                "test_date": test_date
            }
        else:
            date = {'train_date': None, 'val_date': None, 'test_date': None}

        # if len(y_train_3d.shape) == 3 and y_train_3d.shape[-1] == 1:
        # y_train_3d = y_train_3d.reshape(y_train_3d.shape[0], y_train_3d.shape[1])
        # y_val_3d = y_val_3d.reshape(y_val_3d.shape[0], y_val_3d.shape[1])
        # y_test_3d = y_test_3d.reshape(y_test_3d.shape[0], y_test_3d.shape[1])

        # if self.prepro_conf["if_multiple"] == False :
        # 20240229 seongyeop 수정
        # y_train_3d = y_train_3d.reshape(-1, 1)  # Ensuring the right shape
        # y_val_3d = y_val_3d.reshape(-1, 1)
        # y_test_3d = y_test_3d.reshape(-1, 1)
        # if self.prepro_conf["is_classification"]:# 3d shape
        #     y_train_3d = y_train_3d[:,-1]
        #     y_val_3d = y_val_3d[:,-1]
        #     y_test_3d = y_test_3d[:,-1]
        # else:# we only use the last timstep among multiple steps
        #     y_train_3d = y_train_3d[:,-1:]
        #     y_val_3d = y_val_3d[:,-1:]
        #     y_test_3d = y_test_3d[:,-1:]
        # else:
        #     y_train_3d = y_train_3d[:,-1:]# 2d shape
        #     y_val_3d = y_val_3d[:,-1:]
        #     y_test_3d = y_test_3d[:,-1:]
        # else:
        #     raise Exception("we don't support multiple steps forward prediction yet")

        input_size = x_train_3d.shape[
            1:
        ]  # Shape should be something like (n_hist, features)
        # print(len(s_y_train.shape))
        # if len(y_train_3d.shape) > 1:
        #     out = y_train_3d.shape[1]
        # print(s_y_train.shape[1])
        # else:
        # raise ValueError("something is wrong with 'y' data.")
        if self.prepro_conf["is_classification"]:
            out = num_classes
        else:
            out = 1

        if self.prepro_conf["verbose"]:
            print(X_3d.shape, y_3d.shape)
            print(
                f"x_train_3d shape: {x_train_3d.shape}, y_train_3d shape: {y_train_3d.shape}"
            )
            print(f"x_val_3d shape: {x_val_3d.shape}, y_val_3d shape: {y_val_3d.shape}")
            print(
                f"x_test_3d shape: {x_test_3d.shape}, y_test_3d shape: {y_test_3d.shape}"
            )
            print(f"input_size: {input_size}, out: {out} ")
            print("Scaler fitted on:", self.x_scaling.n_features_in_, "features")
            if self.prepro_conf["is_classification"]:
                print("Classification is True")
            else:
                print("Scaler fitted on:", self.y_scaling.n_features_in_, "features")

            for x, y in zip(X[:train_end_idx][:6], y[:train_end_idx][:6]):
                print(f"{x} [{y}]")
            num_samples = 1

            def print_formatted_data(x_data, y_data, num_samples=num_samples):
                for i in range(num_samples):
                    x_sample = x_data[i]
                    y_sample = y_data[i]
                    print("X:", x_sample)
                    print("Y:", y_sample if y_sample.size else "[]")

            print("Train Samples:")
            print_formatted_data(x_train_3d, y_train_3d)

            print("Validation Samples:")
            print_formatted_data(x_val_3d, y_val_3d)

            print("Test Samples:")
            print_formatted_data(x_test_3d, y_test_3d)


        scaled_data = {
            "s_x_train": x_train_3d,
            "s_y_train": y_train_3d,
            "s_x_val": x_val_3d,
            "s_y_val": y_val_3d,
            "s_x_test": x_test_3d,
            "s_y_test": y_test_3d,
            "input_size": input_size,
            "out": out,
            "scaler_x": self.x_scaling,
            "scaler_y": self.y_scaling,
        }

        return scaled_data, date

    def remove_outliers_iqr(self, df, target_column):

        # Calculating the 1st and 3rd quartiles
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1

        # Defining bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtering the DataFrame to remove outliers
        filtered_df = df[
            (df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)
        ]  # 순서가 망가짐.

        return filtered_df

    def process_data_classification(
        self, df, target_column, drop_columns=None, random_shuffle=True
    ):

        def classify_columns(df):
            int_float_nan_cols = []
            other_cols = []

            for col in df.columns:
                # Check if column contains only int, float, and NaN values
                if (
                    df[col]
                    .apply(lambda x: isinstance(x, (int, float)) or pd.isna(x))
                    .all()
                ):
                    int_float_nan_cols.append(col)
                else:
                    other_cols.append(col)

            return int_float_nan_cols, other_cols

        int_float_nan_cols, other_cols = classify_columns(df)

        if self.prepro_conf["is_classification"]:
            if target_column not in int_float_nan_cols:
                int_float_nan_cols = list(int_float_nan_cols) + [target_column]
        # print(f"self.data_pre_config['classification']: {self.data_pre_config['classification']}")
        # print(f"int_float_nan_cols: {int_float_nan_cols}")
        full_data = df[int_float_nan_cols]
        full_data = full_data.dropna()
        # print(f"len of original df: {len(df)}, after dropping nan : {len(full_data)}")
        # print(full_data)
        # Drop unnecessary columns
        if drop_columns:
            full_data.drop(drop_columns, axis=1, inplace=True)

        counts = full_data[target_column].value_counts()
        print(counts)
        # Map target column to numerical values
        unique_classes = full_data[target_column].unique()
        sorted_labels = np.sort(unique_classes)
        label_mapping = {original: new for new, original in enumerate(sorted_labels)}
        full_data[target_column] = full_data[target_column].replace(label_mapping)
        number_of_classes = len(unique_classes)

        # samples_per_class = max(len(full_data) // number_of_classes, 1)

        # Get string columns
        string_columns = full_data.select_dtypes(include="object").columns

        # Initialize a dictionary to store the reverse mappings
        reverse_label_mappings = {}
        pd.options.mode.chained_assignment = None  # default='warn'
        # Convert each string column to numerical values and store the reverse mapping
        for col in string_columns:
            # Convert to category and get codes
            full_data[col] = full_data[col].astype("category")
            # Store the reverse mapping for this column
            reverse_label_mappings[col] = dict(enumerate(full_data[col].cat.categories))
            # Now update the DataFrame with the codes
            full_data[col] = full_data[col].cat.codes.astype("int64")

        # Calculate the smallest class size
        print(full_data[target_column])
        smallest_class_size = full_data[target_column].value_counts().min()
        print("smallest_class_size", smallest_class_size)
        # Calculate the maximum allowed size for any class (10% more than the smallest class size)
        max_allowed_size = int(smallest_class_size * 1.2)
        # Initialize an empty DataFrame for the balanced dataset
        balanced_df = pd.DataFrame()
        # Iterate through each unique class in the target column
        for class_label in full_data[target_column].unique():
            # Filter the DataFrame for the current class
            df_class = full_data[full_data[target_column] == class_label]
            # Check if the current class size exceeds the maximum allowed size
            if len(df_class) > max_allowed_size:
                # If so, randomly sample down to the maximum allowed size
                df_class_sampled = df_class.sample(n=max_allowed_size, random_state=42)
            else:
                # Otherwise, keep all samples from the current class
                df_class_sampled = df_class
            # Append the processed class DataFrame to the balanced DataFrame
            balanced_df = pd.concat([balanced_df, df_class_sampled], ignore_index=True)

        print(balanced_df[target_column].value_counts())

        # Shuffle data if required
        if random_shuffle:
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
                drop=True
            )
            full_data = balanced_df[: len(full_data)]
        else:
            full_data = full_data[: len(full_data)]

        print(f"random_shuffle: {random_shuffle}")
        print(f"mapped dicts: {reverse_label_mappings}")

        return full_data, label_mapping, reverse_label_mappings, number_of_classes

    # example usage data, reverse_label_mappings = process_data_classification(full_data, target_column, drop_columns=None, random_shuffle=True)

    @staticmethod
    def check_date_columns(df):

        date_columns = []

        def is_date_format(s):
            pattern = (
                r"^(\d{4}[-/:]\d{1,2}[-/:]\d{1,2})|(\d{1,2}[-/:]\d{1,2}[-/:]\d{4})"
            )
            if isinstance(s, str):
                import re

                return bool(re.match(pattern, s.split(" ")[0]))
            else:
                return False

        for column in df.columns:
            # Track if any date formats are found within the column
            has_date_format = False
            for value in df[column].dropna().unique():
                if is_date_format(str(value)):
                    has_date_format = True
                    break  # Stop checking this column if a date format is found

            if has_date_format:
                try:
                    # Attempt to convert column to datetime if a date format is found
                    converted_column = pd.to_datetime(df[column], errors="coerce")
                    if pd.api.types.is_datetime64_any_dtype(converted_column):
                        date_columns.append(column)
                        print(f"Datetime column: {column}")
                except Exception as e:
                    print(f"Error processing column {column}: {e}")
                    continue
        return date_columns


def worker_init_fn(worker_id=0):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def dataloader_classification(x, y, batch_size, shuffle=True):
    dataset = Dataseter(torch.FloatTensor(x), torch.LongTensor(y))
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
    )
    return loader


def dataloader_regression(x, y, batch_size, shuffle=False):
    dataset = Dataseter(torch.FloatTensor(x), torch.FloatTensor(y))
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn,
    )
    return loader


def create_3d_data(X, y, is_classification, back_steps=1, fw_steps=0):
    Xs, ys = [], []
    if back_steps == 0:  # 지금 사용 안 하는 것 같은데,
        return X[:, None, :], y[:, None]  # Add channel dimension
    else:
        for i in range(len(X) - back_steps - fw_steps + 1):
            Xs.append(X[i : (i + back_steps)])
            # if fw_steps > 0:
            #     ys.append(y[(i + back_steps) : (i + back_steps + fw_steps)])
            # else:
            if is_classification:
                ys.append(y[i + back_steps + fw_steps - 1])
            else:
                ys.append(y[i + back_steps + fw_steps - 1, None])

        return np.array(Xs), np.array(ys)


def to3d(X, back_steps=1, fw_steps=0):
    Xs = []
    for i in range(len(X) - back_steps - fw_steps + 1):
        Xs.append(X[i : (i + back_steps)])
    return np.array(Xs)


# def calculate_3d_shape(
#     full_train_data_number, n_hist=1, fw_steps=0, forward_multiple=False
# ):
#     Xs_shape = (full_train_data_number - n_hist - fw_steps + 1, n_hist)
#     if forward_multiple:
#         ys_shape = (full_train_data_number - n_hist - fw_steps + 1, fw_steps)
#     else:
#         ys_shape = (full_train_data_number - n_hist - fw_steps + 1,)
#     return Xs_shape, ys_shape


# Define the custom Dataset class
class Dataseter(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels
        print("====x====")
        print(self.X.shape)
        print("====y====")
        print(self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def validate_data_parameters(n_hist, fw_steps, if_mltfwd):
    if n_hist <= 0:
        raise ValueError("n_hist must be greater than 0")

    if fw_steps == 0 and if_mltfwd:
        print("Setting if_mltfwd to False as fw_steps is 0")
        if_mltfwd = False
    return n_hist, fw_steps, if_mltfwd


# Assuming df is your DataFrame
def drop_index_column(df):
    # Check for common index column names and drop them
    common_index_column_names = ["index", "Unnamed: 0"]
    for column in common_index_column_names:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
            return df  # Assuming only one such column exists

    # If no common names found, check for a column that exactly matches the DataFrame index
    for column in df.columns:
        if df[column].is_monotonic_increasing and (df[column].dtype == "int64"):
            # Check if the column is a sequence starting from 0 or 1
            if df[column].iloc[0] in [0, 1] and df[column].equals(
                pd.RangeIndex(
                    start=df[column].iloc[0], stop=df[column].iloc[0] + len(df)
                )
            ):
                df.drop(column, axis=1, inplace=True)
                return df

    return df  # Return the DataFrame if no index column is found or after dropping it


def check_std_of_df(df, target):
    date_columns = RawDataProcessor.check_date_columns(df)
    df_no_target_date = df.drop(columns=[target] + date_columns, axis=1)
    columns_with_deviation = []
    columns_with_error = []

    for column in df_no_target_date.columns:
        try:
            # Only consider the column if its standard deviation can be calculated,
            # which implies it's numeric.
            std_dev = df_no_target_date[column].std()
            if std_dev > 0:
                columns_with_deviation.append(column)
        except TypeError:
            # If a TypeError occurs (cannot convert string to float), append to a different list
            columns_with_error.append(column)

    if columns_with_deviation:
        return columns_with_deviation + columns_with_error
    else:
        return 0


def encode_string_columns(full_data):
    # Select only columns of object type
    object_cols = full_data.select_dtypes(include=["object"]).columns
    reverse_label_mappings = {}
    for col in object_cols:
        # Convert to category and get codes
        full_data[col] = full_data[col].astype("category")
        # Store the reverse mapping for this column
        reverse_label_mappings[col] = dict(enumerate(full_data[col].cat.categories))
        # Now update the DataFrame with the codes
        full_data[col] = full_data[col].cat.codes.astype("int64")

    return full_data, reverse_label_mappings
