import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')
from ordinal_encoding import ordinal_encoder
model = joblib.load(r"models/RF_model.joblib")


# Columns - Categorical
opt_week_day=['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
opt_driver_age=['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
opt_driver_experience=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
opt_vehicle = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Public (13-45 seats)',
       'Lorry (11-40Q)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj',
       'Bicycle']

opt_accident_occured_area=['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Rural village areasOffice areas', 'Unknown',
       'Recreational areas']

opt_junction_type=['Y Shape', 'No junction', 'Crossing', 'Other', 'Unknown', 'O Shape',
       'T Shape', 'X Shape']

opt_road_type=['Asphalt roads', 'Earth roads', 'Gravel roads', 'Other',
       'Asphalt roads with some distress']

opt_road_condition=['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
opt_light_condition=['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
       'Darkness - lights unlit']

features = ['Hour', 'Day_of_week', 'Age_band_of_driver', 'Area_accident_occured',
       'Type_of_vehicle', 'Types_of_Junction', 'Driving_experience',
       'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
       'Number_of_vehicles_involved', 'Number_of_casualties']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)

with st.form("Prediction From"):
    st.subheader("Enter the inputs")

    c1,c2=st.columns(2)
    with c1:

        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        day_of_week=st.selectbox("Day of the week",options=opt_week_day)
        driver_age=st.selectbox("Age of the Driver",options=opt_driver_age)
        type_of_vehicle=st.selectbox("Type of vehicle",options=opt_vehicle)
        number_of_vehicles_involved=st.slider("Number of Vehicles Involved",1,10,value=1,format="%d")
        type_road=st.selectbox("Type of Road",options=opt_road_type)
    
    with c2:
        condition_of_Road=st.selectbox("Condition of the Road",options=opt_road_condition)
       
        casualties=st.slider("Number of casualities",0,10,value=0,format="%d")
        driver_experience=st.selectbox("Experience of the Driver",options=opt_driver_experience)
        accident_occured_area=st.selectbox("Area where accident occured",options=opt_accident_occured_area)
        junction_type=st.selectbox("Type of Junction",options=opt_junction_type)
        light_condition=st.selectbox("Light condition",options=opt_light_condition)
    
    submit = st.form_submit_button("Predict")
if submit:

    day_of_week=ordinal_encoder(day_of_week,opt_week_day)
    driver_age=ordinal_encoder(driver_age,opt_driver_age)
    accident_occured_area=ordinal_encoder(accident_occured_area,opt_accident_occured_area)
    type_of_vehicle=ordinal_encoder(type_of_vehicle,opt_vehicle)
    driver_experience=ordinal_encoder(driver_experience,opt_driver_experience)
    type_road=ordinal_encoder(type_road,opt_road_type)
    condition_of_Road=ordinal_encoder(condition_of_Road,opt_road_condition)
    light_condition=ordinal_encoder(light_condition,opt_light_condition)
    junction_type=ordinal_encoder(junction_type,opt_junction_type)
    
    data=np.array([day_of_week,driver_age,accident_occured_area,type_of_vehicle,junction_type,
              driver_experience,type_road,condition_of_Road,light_condition,
              number_of_vehicles_involved,casualties,hour]).reshape(1,-1)

    pred=model.predict(data)

    st.title(pred[0])
