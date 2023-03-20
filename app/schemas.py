from pydantic import BaseModel
from enum import Enum

import datetime

class WeatherType(str, Enum):
    sunny = "Sunny"
    windy = "Windy"
    cloudy = 'Cloudy'
    fog = 'Fog'
    stormy = 'Stormy'
    sandstorm = 'Sandstorms'
    
class Traffic(str, Enum):
    high = "High"
    jam = "Jam"
    low = 'Low'
    med = 'Medium'
    
class OrderType(str, Enum):
    buffet = 'Buffet'
    drinks = 'Drinks'
    meal = 'Meal'
    snack = 'Snack'
    
class VehicleType(str, Enum):
    bicycle = 'bicycle'
    electric_scooter = 'electric_scooter'
    motorcycle = 'motorcycle'
    scooter = 'scooter'

class Fest(str, Enum):
    yes = "Yes"
    no = "No"
    
class CityType(str, Enum):
    metro = "Metropoliton"
    urban = "Urban"
    semi_urban = "Semi-Urban"

class Data(BaseModel):
    Delivery_person_ID: str
    Delivery_person_Age: int
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: datetime.date
    Time_Orderd: datetime.time
    Time_Order_picked: datetime.time
    Weather:WeatherType
    Road_traffic_density: Traffic
    Vehicle_condition: int
    Type_of_order: OrderType
    Type_of_vehicle: VehicleType
    multiple_deliveries: int
    Festival: Fest
    City: CityType