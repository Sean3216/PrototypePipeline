from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Boruta:

    def __init__(self, boruta_conf, prepro_conf, x_value=None):
        self.boruta_conf = boruta_conf
        self.prepro_conf = prepro_conf
        self.x_value = x_value

    def run_boruta(self, cleaned_df, target_variable):

        X = cleaned_df.drop(columns=[target_variable])  # features
        y = cleaned_df[target_variable]  # target variable
        # findx = X.values  # features for inference
        # findy = y.values  # target for inference
        findx = X[: self.boruta_conf["data_num"]].values  # features for inference
        findy = y[: self.boruta_conf["data_num"]].values  # target for inference
        if self.x_value is not None:
            selected_features = self.x_value
        else:
            if self.prepro_conf["is_classification"] == True:
                # boruta_selector = BorutaPy(
                #     RandomForestClassifier(),
                #     n_estimators="auto",
                #     verbose=self.boruta_conf['verbose'],
                #     random_state=42,
                #     perc=self.boruta_conf['perc'],
                #     max_iter=self.boruta_conf['max_init'],
                #     )
                boruta_selector = BorutaPy(
                    RandomForestClassifier(max_depth=self.boruta_conf["max_depth"]),
                    n_estimators=self.boruta_conf["n_estimators"],
                    verbose=self.boruta_conf["verbose"],
                    random_state=42,
                    perc=self.boruta_conf["perc"],
                    max_iter=self.boruta_conf["max_iter"],
                )
            # Create a Boruta object with RandomForestRegressor
            else:

                boruta_selector = BorutaPy(
                    RandomForestRegressor(max_depth=self.boruta_conf["max_depth"]),
                    n_estimators=self.boruta_conf["n_estimators"],
                    verbose=self.boruta_conf["verbose"],
                    random_state=42,
                    perc=self.boruta_conf["perc"],
                    max_iter=self.boruta_conf["max_iter"],
                )
            boruta_selector.fit(findx, findy)
            # Get selected features
            selected_features = X.columns[boruta_selector.support_]

        if len(selected_features) == 0:
            print(
                "feature_columns selected is empty, proceeding with all columns in the DataFrame"
            )
            selected_features = cleaned_df.columns

        return selected_features
