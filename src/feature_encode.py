import pandas as pd
import numpy as np

import os
from collections import defaultdict
import copy
import joblib

import config

def label_encode(df):
    df = copy.deepcopy(df)

    traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3, "NONE": None}
    fest_map = {"Yes": 1, "No": 0, 'NONE': None}
    city_map = {"Semi-Urban": 0, "Urban": 1, "Metropolitian": 2, 'NONE': None}

    df["City"] = df["City"].map(city_map)
    df["Festival"] = df["Festival"].map(fest_map)
    df["Road_traffic_density"] = df["Road_traffic_density"].map(traffic_map)

    return df

def target_mean_encode(df, cols):
    df = copy.deepcopy(df)

    encoded_df_lst = []
    mappers = defaultdict(list)

    for fold in range(5):
        df_train = df[df.kfold!=fold].reset_index(drop=True)
        df_valid = df[df.kfold==fold].reset_index(drop=True)

        for col in cols:
            col_map = df_train.groupby(col)['Time_taken'].mean().to_dict()

            df_valid[col] = df_valid[col].map(col_map)

            mappers[col].append(col_map)
        
        encoded_df_lst.append(df_valid)

    encoded_df = pd.concat(encoded_df_lst, axis=0)

    return encoded_df, mappers


def encode_features(df):

    df = label_encode(df)

    cols = df.select_dtypes(include='object').columns.values

    df, encoders = target_mean_encode(df, cols)

    return df, encoders

def col_agg_encoder(col_encoder):
    agg_dct = defaultdict(list)

    for item in col_encoder:
        for key,val in item.items():
            agg_dct[key].append(val)

    mean_dct = {}
    for key,val in agg_dct.items():
        mean_dct[key] = np.mean(val)

    return mean_dct

if __name__ == "__main__":
    df = pd.read_csv(config.CMB_CAT_DATA_PATH)

    df, encoder_map = encode_features(df)

    agg_encoder_map = {}
    for col in encoder_map:
        agg_encoder_map[col] = col_agg_encoder(encoder_map[col])

    df.to_csv(config.NUM_DATA_PATH, index=False)

    joblib.dump(agg_encoder_map, os.path.join(config.MODEL_PATH, "target_mean_mapper.bin"))