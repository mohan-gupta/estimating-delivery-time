import numpy as np
import pandas as pd

import pygeohash as pgh

from functools import reduce
import itertools
import datetime
import copy

import config

def fix_time(x):
    """Function to fix time
    x is a string of time-> hh:mm or h:mm"""
    if type(x)==float:
        return np.nan
    
    if x=='24:00' or x=='23:60':
        return "0:0"
    
    if x[:2]=="24":
        return "00:"+x[3:]
    
    if x[-2]=='6':
        if x[1]==':':
            return str(int(x[0])+1)+":0"+x[-1]
        return str((int(x[:2])+1)%24)+":0"+x[-1]
    return x

def ratings(x):
    """Function to fix ratings between 1 to 5"""
    if x<1:
        return 1
    elif x>5:
        return 5
    return x

#----------------Date Time----------------
def bin_time(hour):
    """Function to divide the day based on hours"""
    if hour>=2 and hour<6:
        return "dawn"
    elif hour >= 6 and hour<10:
        return "morning"
    elif hour >= 10 and hour<14:
        return "noon"
    elif hour >=14 and hour <18:
        return "afternoon"
    elif hour >= 18 and hour<22:
        return "evening"
    else:
        return "midnight"

def compute_time_ftrs(df):
    """Calculates the difference between order pickup time and time ordered in minutes
    And extracts hour and minute from the order and pickup time"""
    df = copy.deepcopy(df)
    df["Time_Orderd"] = df["Time_Orderd"].apply(lambda x: datetime.strptime(x, "%H:%M") if x != "NONE" else x)
    df["Time_Order_picked"] = df["Time_Order_picked"].apply(lambda x: datetime.strptime(x, "%H:%M"))
    
    df.loc[:, "pickup_delay"] = -1
    
    for idx in range(len(df)):
        time_ordered = df.loc[idx, "Time_Orderd"]
        if time_ordered == "NONE":
            df.loc[idx, "pickup_delay"] = None
        else:
            time_order_picked = df.loc[idx, "Time_Order_picked"]
            delay_sec = (time_order_picked - time_ordered).seconds
            df.loc[idx, "pickup_delay"] = delay_sec/60
    
    df.loc[:, "order_hour"] = df["Time_Order_picked"].apply(lambda x: x.hour)
    df.loc[:, "order_minute"] = df["Time_Orderd"].apply(lambda x: None if type(x)==str else x.minute)
    df.loc[:, "order_minute_picked"] = df["Time_Order_picked"].apply(lambda x: x.minute)

    df.loc[:, "time_of_day"] = df["order_hour"].apply(bin_time)
    
    df = df.drop(["Time_Orderd", "Time_Order_picked"], axis=1)
    
    return df


def extract_date_ftrs(df):
    """Extracts month, day of week, week of year, weekend from the data"""
    df = copy.deepcopy(df)

    order_date = pd.to_datetime(df['Order_Date'], dayfirst=True)

    df.loc[:, "month"] = order_date.apply(lambda x: x.month)
    df.loc[:, "day"] = order_date.apply(lambda x: x.day)
    df.loc[:, 'day_of_week'] = order_date.apply(lambda x: x.dayofweek)
    df.loc[:, 'week_of_year'] = order_date.apply(lambda x: x.weekofyear)
    df.loc[:, 'weekend'] = order_date.apply(lambda x: x.dayofweek>=5).astype(np.int8)

    return df

#------------------Latitudes and Longitudes------------------
def apply_geohashing(df):
    """Applies Geohashing on the locations(latitudes and longitudes)"""
    df = copy.deepcopy(df)
    df['restaurant_geohash'] = df.apply(lambda x: pgh.encode(
                                                x.Restaurant_latitude, x.Restaurant_longitude, precision=5
                                            ), axis=1)

    df['delivery_loc_geohash'] = df.apply(lambda x: pgh.encode(
                                                    x.Delivery_location_latitude, x.Delivery_location_longitude, precision=5
                                                ), axis=1)
    
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

def manhattan_distance_lat_long(lat1, lng1, lat2, lng2):
    """Computes manhattan distance between two locations"""
    lat_dist = haversine_dist(lat1, lng1, lat1, lng2)
    lng_dist = haversine_dist(lat1, lng1, lat2, lng1)
    return lat_dist + lng_dist

def bearing_dist(lat1, lng1, lat2, lng2):
    """Computes Bearing distance between two locations"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_dif_rad = np.radians(lng2 - lng1)
    
    lat1, lng1, lat2, lng2 = np.radians([lat1, lng1, lat2, lng2])
    
    y = np.sin(lng_dif_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_dif_rad)
    
    return np.degrees(np.arctan2(y, x))

def compute_distances(df):
    """Returns the data frame by adding columns for haversine, manhattan,
    and bearing distance."""
    df = copy.deepcopy(df)

    df.loc[:, "haversine"] = df.apply(lambda x:
                                 haversine_dist(
                                     x.Restaurant_latitude, x.Restaurant_longitude,
                                     x.Delivery_location_latitude, x.Delivery_location_longitude
                            ), axis=1)
    
    df.loc[:, "manhattan"] = df.apply(lambda x:
                                 manhattan_distance_lat_long(
                                     x.Restaurant_latitude, x.Restaurant_longitude,
                                     x.Delivery_location_latitude, x.Delivery_location_longitude
                            ), axis=1)
    
    df.loc[:, "bearing"] = df.apply(lambda x:
                                 bearing_dist(
                                     x.Restaurant_latitude, x.Restaurant_longitude,
                                     x.Delivery_location_latitude, x.Delivery_location_longitude
                            ), axis=1)
    
    return df

#-----------------Combining Categories-----------------
def combine_cat(df):
    """Greedily combine categorical columns in the data frame"""
    df = copy.deepcopy(df)

    cat_cols = df.select_dtypes(include="object").columns.values
    combine_cat = cat_cols[2:]

    combinations = itertools.combinations(combine_cat, 2)

    for c1, c2 in combinations:
        df.loc[:, c1+"_"+c2] = df[c1] + "_" + df[c2]
    
    return df

#Apllying Above functions on data
def compose_functions(*functions):
    return reduce(lambda f,g: lambda x: g(f(x)), functions)

def main(df):
    my_func = compose_functions(compute_time_ftrs,  # added time features from order and pickup time
                                extract_date_ftrs,  # added data time features from order date
                                apply_geohashing,   # Added Geo hashes of restaurant and delivery locations
                                compute_distances,  # Added Distance info b/w restaurant and delivery location
                                combine_cat)    # Greedily combined all the categorical features
    
    result = my_func(df)   

    return result

if __name__ == "__main__":
    df = pd.read_csv(config.RAW_DATA_PATH)
    df = main(df)

    df.to_csv(config.CMB_CAT_DATA_PATH, index=False)
