import streamlit as st
from taxipred.utils.helpers import read_api_endpoint
from taxipred.utils.constants import ML_MODEL_PATH
import pandas as pd
import joblib


all_data = read_api_endpoint("all-taxi-rides")
prediction = read_api_endpoint("taxi-price-prediction")

all_data_df = pd.DataFrame(all_data.json())

model = joblib.load(ML_MODEL_PATH)

def main():
    st.markdown("# Taxi Prediction Dashboard")

    tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

    with tab1:
        st.markdown("### All data tab")
        if st.button("Show all data"):
            st.dataframe(all_data_df)

    with tab2: 
        trip_distance_km = st.number_input("Trip distance in km: ", min_value=1, max_value=200, value=10)
        time_of_day = st.selectbox(
            "Time of day: ",
            options=["Morning", "Afternoon", "Evening", "Night"])            
        day_of_week = st.selectbox(
            "Weekday or weekend: ",
            options=["Weekday", "Weekend"]
        )
        passenger_count = st.number_input("Passenger count: ", min_value=1, max_value=4, value=2)        
        traffic_conditions = st.selectbox(
            "Low, medium or high traffic: ",
            options=["Low", "Medium", "High"]
        )      
        weather = st.selectbox(
            "Select weather: ",
            options=["Clear", "Rain", "Snow"]
        )                
        base_Fare = st.number_input("Select base fare (2-5$): ", min_value=2, max_value=10, value=2)            
        per_km_rate = st.number_input("$ per km: ", min_value=0.5, max_value=2.0, value=1.0)
        per_minute_rate = st.number_input("$ per minute: ", min_value=0.1, max_value=1.0, value=0.1)        
        trip_duration_minutes = st.number_input("Trip duration in minutes: ", min_value=5, max_value=200, value=15)

        features_to_model = {
            "Trip_Distance_km": trip_distance_km,
            "Passenger_Count": passenger_count,
            "Base_Fare": base_Fare,
            "Per_Km_Rate": per_km_rate,
            "Per_Minute_Rate": per_minute_rate,
            "Trip_Duration_Minutes": trip_duration_minutes,
            
            # Since we did one hot encoding for these in the df, I will have to do the below
            # Traffic_Conditions (one-hot-encoding)
            "Traffic_Conditions_High": 1 if traffic_conditions == "High" else 0,
            "Traffic_Conditions_Low": 1 if traffic_conditions == "Low" else 0,
            "Traffic_Conditions_Medium": 1 if traffic_conditions == "Medium" else 0,

            # Weather (one-hot-encoding)
            "Weather_Clear": 1 if weather == "Clear" else 0,
            "Weather_Rain": 1 if weather == "Rain" else 0,
            "Weather_Snow": 1 if weather == "Snow" else 0,
            
            # Time_of_Day (one-hot-encoding)
            "Time_of_Day_Afternoon": 1 if time_of_day == "Afternoon" else 0,
            "Time_of_Day_Evening": 1 if time_of_day == "Evening" else 0,
            "Time_of_Day_Morning": 1 if time_of_day == "Morning" else 0,
            "Time_of_Day_Night": 1 if time_of_day == "Night" else 0,
            
            # Day_of_Week (one-hot-encoding)
            "Day_of_Week_Weekday": 1 if day_of_week == "Weekday" else 0,
            "Day_of_Week_Weekend": 1 if day_of_week == "Weekend" else 0    
        }

        if st.button("Make prediction: "):
            X = pd.DataFrame([features_to_model])
            y_pred = model.predict(X)
            st.write(y_pred[0])

    with tab3:
        st.write("Hej")




if __name__ == "__main__":
    main()
