"""
    regression : 
    - input : 2d or 3d (batch, n_hist, features)
    - output : 2d (batch, 1)
    - n_hist >= 1
    - fw_steps >= 0
    
    classification :
    - input : 2d or 3d (batch, n_hist, features)
    - output : 1d (batch, ) for crossentropy loss
    - n_hist >= 1
    - fw_steps >= 0
"""

prepro_conf = {
    "n_hist": 1,  # number of historical steps used for prediction if 1 then only current step is used | if 3 then current and 2 previous steps are used x = [x-2, x-1, x]
    "fw_steps": 0,  # number of steps to predict forward if prediction is in same row then 0, if next row then 1 | if 3 then y = [y+1, y+2, y+3] or y = [y+fw_steps] if if_mltfwd = False
    "train_val_test_ratio": (0.7, 0.15),  # train, validation, test ratio
    "full_train_data_number": 10000,  # number of rows to process, | first N datas to see
    "scaler_name": "standard",  # "standard" or "minmax" or "robust"
    "if_multiple": False,  # if True then predict multiple steps forward, if False then predict only 1 step forward
    "shuffle": True,
    "combine_val_train": False,  # if True then validation set is combined with training set
    "verbose": False,  # if True then print data shapes and other information
    "apply_scaler": True,
    "change3d": True
}
seed_config = {"set_seed": True, "seed": 0}

boruta_conf = {
    "n_estimators": "auto",  # number of trees to use for feature selection
    "n_feat_plot": 2,  #
    "data_num": 5000,  # number of features to check to get feature importance
    "max_depth": 2,
    "max_iter": 5,
    "perc": 50,
    "verbose": 1,  # if 0 then no print | if 1 then print progress | if 2 then print all
}

bayesian_conf = {
    "init_points": 1,
    "n_iter": 2,
    "if_penalize_std": True,
    "std_penalty_factor": 10,
    "similarity_percentage": 0.3,
    "verbose": 1,  # if 0 then no print | if 1 then print progress | if 2 then print all
    "if_score_on_val": True,  # if True then score is calculated on validation set | if False then score is calculated on test set
}

train_conf = {
    "num_epochs": 2,  # number of epochs to train
    "patience": 15,
    "early_stop": True,
    "verbose": False,
    "loss_function_name": "mse",  # "rmse", "standard", "crossentropy", "mse", "mae", "huber", "smoothL1Loss", "binary"  | "standard", "rmsle" is custom loss function
    "loss_alpha": 5,  # alpha for loss function (loss = alpha * loss_mse + loss_std)
    "learning_rate": 0.001,  # learning rate for optimizer
    "weight_decay": 0.001,  # if > 0 then L2 regularization is used | if = 0 or None then no regularization is used
    "betas": (
        0.9,
        0.999,
    ),  # betas for Adam optimizer, standard values (0.9, 0.999) | betas=(0.0, 0.0) for SGD optimizer | if (0.5, 0.999) then SGD with momentum | if (0.9, 0.0) then Adam with momentum
    "l1_regularization": False,  # if True then L1 regularization is used | if False then no regularization is used
    "l1_alpha": 0.0001,  # alpha for L1 regularization | loss=loss_mse + alpha * L1 | effect is to make weights sparse (close to 0)
    "if_plot_every_epoch": False,  # if True then plot train and validation loss after every epoch
    "save_plot_as_image": False,  # if True then save plot as image | if False then do not save plot as image
    "plot_start": 0,  # start index for plot
    "plot_end": 10000,  # end index for plot
    "save_parameter_info_as_file": False,
}

model_conf = {
    "base_batch_size": 128,  # base batch size for training
    "activation_name": "relu",  # activation function for hidden layers | activation_dict["leaky_relu"] or activation_dict["relu"] or activation_dict["tanh"] or activation_dict["gelu"]
    "dropout": 0.2,  # dropout see utils.model_trainers.py
    "normalization": False,  # if True then normalization is used | if False then no normalization is used
    "normalization_name": "batch",  # "batch" or "instance" normalization
    "init_verbose": False,  # if True then print model summary | if False then do not print model summary
}

# "CNN1D", "MLP", "LSTM", "CNNLSTM", "NBEATS", "NHITS", "NLINEAR", "TSTMODEL", "TFTMODEL", "BiLSTM", "WeightAdaptingNN"
select_model = {
    "model_name": "MLP",  # NLINEAR, TST, TFT no norm
}
ensemble_conf = {
    "models": [
        "CNN1D",
        "MLP",
        "LSTM",
        "CNNLSTM",
        "NBEATS",
        "NHITS",
        "NLINEAR",
        "TSTMODEL",
        "TFTMODEL",
        "BiLSTM",
        "WeightAdaptingNN",
    ],
    "regression_models": [
        #"CNN1D",
        "MLP",
       ## "LSTM",
        # "CNNLSTM",
        "NBEATS",
        #"NHITS",
        # "NLINEAR",
       ## "TSTMODEL",
       ## "TFTMODEL",
        # "BiLSTM",
       # "WeightAdaptingNN",
    ],
    "classification_models": [
        # "CNN1D",
        "MLP",
        # "LSTM",
        # "CNNLSTM",
        "NBEATS",
        #"NHITS",
        # "NLINEAR",
        # "TSTMODEL",
       ## "TFTMODEL",
       # "BiLSTM",
       # "WeightAdaptingNN",
    ],
    # "models": ["CNN1D", "MLP", "LSTM"],
    "best_few_n": 2,
}
# "ensemble_models": ["CNN1D", "MLP", "LSTM", "CNNLSTM", "NBEATS", "NHITS", "NLINEAR", "TSTMODEL", "TFTMODEL", "BiLSTM", "WeightAdaptingNN"]}

# Manually define the hyperparameters
pre_hyperparameters = {
    "MLP": {"hidden_size": 256, "layers": 3},
    "CNN1D": {"hidden_size": 128, "layers": 4, "kernel_size": 3},
    "LSTM": {"hidden_size": 256, "layers": 1},
    "CNNLSTM": {"hidden_size": 128, "layers": 3, "kernel_size": 3},
    "NBEATS": {"blocks": 4, "hidden_size": 75, "layers": 3, "theta": 45},
    "NHITS": {"blocks": 4, "hidden_size": 75, "layers": 3, "theta": 45},
    "NLINEAR": {"hidden_size": 128, "layers": 3},
    "TSTMODEL": {"hidden_size": 75, "layers": 3, "n_heads": 8},
    "TFTMODEL": {"hidden_size": 75, "layers": 3, "n_heads": 8},
    "BiLSTM": {"hidden_size": 128, "layers": 2},
    "WeightAdaptingNN": {"hidden_size": 150, "layers": 3},
}
pbounds = {
    "lr": (0.00001, 0.001),
    "hidden_size": (32, 256),
    "layers": (1, 5),
}
