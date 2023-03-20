import argparse

import joblib
import os

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

import config
import model_dispatcher

def run_folds(fold, model, df):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["Time_taken", "kfold"], axis=1)
    y_train = df_train['Time_taken'].values

    x_valid = df_valid.drop(["Time_taken", "kfold"], axis=1)
    y_valid = df_valid['Time_taken'].values

    reg = model_dispatcher.model[model]
    
    if model == 'catb' or model == 'tuned_catb':
        cat_cols = x_train.select_dtypes(include='object').columns.values
        reg.fit(x_train, y_train, cat_features=cat_cols)
    else:
        reg.fit(x_train, y_train)

    preds = reg.predict(x_valid)

    mse = mean_squared_error(y_valid, preds)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_valid, preds)

    print(f"Fold={fold}, R2 score={r2} and RMSE={rmse}")

    joblib.dump(reg, os.path.join(config.MODEL_PATH, f"{model}_{fold}.bin"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str)

    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    data_dict={'cat':config.RAW_DATA_PATH, 'num':config.CLN_DATA_PATH, 'imp': config.IMPUTED_DATA_PATH}
    
    df = pd.read_csv(data_dict[args.data])

    for fold in range(5):
        run_folds(fold, args.model, df)