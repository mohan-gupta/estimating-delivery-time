import os
import copy
import joblib
from collections import defaultdict

import pygeohash as pgh

import numpy as np
import pandas as pd

MODEL_PATH = "../model"
MAPPER = joblib.load(os.path.join(MODEL_PATH, "target_mean_mapper.bin"))

def label_encode(df):
    df = copy.deepcopy(df)
    traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3, "NONE": None}
    fest_map = {"Yes": 1, "No": 0, 'NONE': None}
    city_map = {"Semi-Urban": 0, "Urban": 1, "Metropolitian": 2, 'NONE': None}

    df["City"] = df["City"].map(city_map)
    df["Festival"] = df["Festival"].map(fest_map)
    df["Road_traffic_density"] = df["Road_traffic_density"].map(traffic_map)

    return df

def haversine_dist(lat1, lng1, lat2, lng2):
    """Computes haversine distance between two locations"""
    AVG_EARTH_RADIUS = 6371  # in km
    lat1, lng1, lat2, lng2 = np.radians([lat1, lng1, lat2, lng2])
    
    lat_diff = lat2 - lat1
    lng_diff = lng2 - lng1
    
    d = (np.sin(lat_diff * 0.5)**2) + (np.cos(lat1) * np.cos(lat2) * np.sin(lng_diff * 0.5) ** 2)
    
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    
    return h

def location_data(df):
    """Applies Geohashing on the locations(latitudes and longitudes)"""
    df = copy.deepcopy(df)
    df['restaurant_geohash'] = df.apply(lambda x: pgh.encode(
                                                x.Restaurant_latitude, x.Restaurant_longitude, precision=5
                                            ), axis=1)

    df['delivery_loc_geohash'] = df.apply(lambda x: pgh.encode(
                                                    x.Delivery_location_latitude, x.Delivery_location_longitude, precision=5
                                                ), axis=1)
    
    df["haversine"] = df.apply(lambda x:
                                 haversine_dist(
                                     x.Restaurant_latitude, x.Restaurant_longitude,
                                     x.Delivery_location_latitude, x.Delivery_location_longitude
                            ), axis=1)
    
    return df

def combine_data(df):
    df = copy.deepcopy(df)
    comb_lst = [('City', 'restaurant_geohash'), ('Festival', 'City'), ('Road_traffic_density', 'City'),
                ('Road_traffic_density', 'Festival'), ('Road_traffic_density', 'Type_of_vehicle'),
                ('Road_traffic_density', 'delivery_loc_geohash'), ('Road_traffic_density', 'restaurant_geohash'),
                ('Type_of_order', 'restaurant_geohash'), ('restaurant_geohash', 'delivery_loc_geohash')]
    
    for c1,c2 in comb_lst:
        df.loc[:, c1+"_"+c2] = df[c1] + "_" + df[c2]
        
    return df

def get_xgb_data(df):
    df = copy.deepcopy(df)
    xgb_data = location_data(df)
    re_cols = ['Weather', 'restaurant_geohash_delivery_loc_geohash',
       'Delivery_person_Age', 'Road_traffic_density_delivery_loc_geohash',
       'Road_traffic_density_Festival', 'Vehicle_condition',
       'Delivery_person_Ratings', 'Road_traffic_density',
       'Road_traffic_density_Type_of_vehicle',
       'Road_traffic_density_City',
       'Road_traffic_density_restaurant_geohash', 'multiple_deliveries',
       'Festival_City', 'Delivery_person_ID', 'haversine',
       'City_restaurant_geohash', 'Type_of_order_restaurant_geohash']
    
    xgb_data = combine_data(xgb_data)
    
    xgb_data = label_encode(xgb_data)
        
    xgb_data = xgb_data.drop(['City', 'Delivery_location_latitude', 'Delivery_location_longitude',
                                      'Festival', 'Order_Date', 'Restaurant_latitude', 'Restaurant_longitude',
                                      'Time_Order_picked', 'Time_Orderd', 'Type_of_order', 'Type_of_vehicle'],
                                     axis=1)
    
    for cat_col in xgb_data.select_dtypes(include='object').columns:
        xgb_data[cat_col] = xgb_data[cat_col].map(MAPPER[cat_col])
    
    return xgb_data[re_cols]

def process_data(data):
    data["Order_Date"] = data["Order_Date"].strftime("%d-%m-%Y")
    data['Time_Orderd'] = data['Time_Orderd'].strftime("%H:%M")
    data['Time_Order_picked'] = data['Time_Order_picked'].strftime("%H:%M")
    
    catboost_data = pd.DataFrame(data=data, index=[1])
    
    xgboost_data = get_xgb_data(catboost_data)
    
    return {
        'cat_data': catboost_data,
        'xgb_data': xgboost_data
    }

def load_models():
    model_dct = defaultdict(list)
    for model_name in ('catb', 'xgb'):
        for fold in range(5):
            file_name = os.path.join(MODEL_PATH, f"tuned_{model_name}_{fold}.bin")
            model = joblib.load(file_name)
            
            model_dct[model_name].append(model)
    
    return model_dct

def model_preds(data, model):
    model_preds = []
    for fold_model in model:
        fold_pred = fold_model.predict(data)
        model_preds.append(fold_pred)
        
    return np.mean(model_preds)

def get_preds(data, model_dct):
    models_data = process_data(data)

    cat_preds = model_preds((models_data['cat_data']), model_dct['catb'])
    xgb_preds = model_preds((models_data['xgb_data']), model_dct['xgb'])
    
    return np.round(np.mean([cat_preds, xgb_preds]), 3)