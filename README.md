# PrototypePipeline
Pipeline to process data and select models from available options (currently XGBoost, Prophet, and DeepAR) to do prediction/forecasting.

### Running the script
The pipeline can be executed with terminal by first navigating to the directory of the scripts and run main.py like the following command:

```shell
python main.py --config [config_dir]
```
Notice that we need to input the directory of the configuration file. Examples of configuration files are located in configs folder.

Current Limitation:
- The current pipeline has not support prediction/forecasting for multiple target/output 
- The current pipeline is tested on 10 datasets (3 classifications data, 7 regression data) using XGBoost and Prophet
1. Regression
   * Air_quality_index from UCI data
   * 전력수요 dataset
   * 물류 data from Dacon
   * Kaggle Maryland Retail Sales Dataset
   * Kaggle Food Demand Warehouse Dataset
   * Kaggle Store Item Demand Dataset
   * KAMP 27 Regression 수주량예측 (T일예정수주량) Dataset
   * KAMP 22 Regression Dataset
   * KAMP 28 Regression Dataset
   * KAMP 49 Regression Dataset
   * KAMP 50 Regression Dataset
   * KAMP 21 Regression Dataset
2. Classification
   * load_churn Dataset
   * load_comp Dataset
   * load_full_data_purchase
   * KAMP 4 Classification Dataset
   * KAMP 8 Classification Dataset
   * KAMP 9 Classification Dataset
   * KAMP 38 Classification Dataset
   * KAMP 3 Classification Dataset
   * KAMP 10 Classification Dataset
   * KAMP 37 Classification Dataset
   * KAMP 35 Classification Dataset
   * KAMP 46 Classification Dataset
   * KAMP 39 Classification Dataset
   * KAMP 14 Classification Dataset
- Further bugs might still exist
- list of data can be seen here (https://docs.google.com/spreadsheets/d/10M-cTBfBbb7LI1mjVSTNXcTgI2pmpvsY/edit#gid=1291101772)
