import streamlit as st
from predict import get_preds, load_models

import base64

MODEL = load_models()

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    add_bg_from_local("assets/bg.jpg")
    title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Estimating Delivery Time.</p>'
    st.markdown(title, unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]:
        person_id = st.text_input("Delivery Person ID", value="MUMRES13DEL03")
        del_lat = st.number_input("Delivery Location Latitude", value=19.208321)
        weather = st.selectbox("Weather", options=('Cloudy', 'Fog', 'Sunny', 'Windy', 'Stormy', 'Sandstorms'))
        mltpl_del = st.selectbox("Multiple Deliveries", options=(0, 1, 2, 3))
    
    with cols[1]:
        person_age = st.number_input("Delivery Person Age", min_value=12, max_value=80, value=26)
        del_lng = st.number_input("Delivery Location Longitude", value=72.864715)
        traffic = st.selectbox("Road Traffic", options=('High', 'Jam', 'Low', 'Medium'))
        fst = st.selectbox("Festival", options=("Yes", "No"))
    
    with cols[2]:
        person_rating = st.number_input("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.5)
        ordr_date = st.date_input(label="Order Date")
        vehicle_cnd = st.selectbox("Vehicle Condition", options=(0, 1, 2, 3))
        city = st.selectbox("City", options=("Metropoliton", "Urban", "Semi-Urban"))
    
    with cols[3]:
        res_lat = st.number_input("Restaurant Latitude", value=19.178321)
        time_ordr = st.time_input("Order Time")
        order_type = st.selectbox("Order Type", options=('Buffet', 'Drinks', 'Meal', 'Snack'))
    
    with cols[4]:
        res_lng = st.number_input("Restaurant Longitude", value=72.834715)
        pickup_time = st.time_input("Pickup Time")
        vehicle_type = st.selectbox("Vehicle Type", options=('bicycle', 'electric_scooter', 'motorcycle', 'scooter'))
    
    if st.columns((2,1,2))[1].button("Predict"):
        data = {"Delivery_person_ID": person_id, "Delivery_person_Age": person_age,
                "Delivery_person_Ratings": person_rating, "Restaurant_latitude": res_lat,
                "Restaurant_longitude": res_lng, "Delivery_location_latitude": del_lat,
                "Delivery_location_longitude": del_lng, "Order_Date": ordr_date,
                "Time_Orderd": time_ordr, "Time_Order_picked": pickup_time,
                "Weather":weather, "Road_traffic_density": traffic,
                "Vehicle_condition": vehicle_cnd, "Type_of_order": order_type,
                "Type_of_vehicle": vehicle_type, "multiple_deliveries": mltpl_del,
                "Festival": fst, "City": city}
        
        pred = get_preds(data, MODEL)
        title = f'<center><p style="font-family:sans-serif; color:white; font-size: 16px; align:center"><span style="background-color:rgba(0,0,0,.6);">The order will be delivered in {pred} minutes.</span></p></center>'
        st.markdown(title, unsafe_allow_html=True)