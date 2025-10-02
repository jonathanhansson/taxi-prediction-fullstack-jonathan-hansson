import streamlit as st
from taxipred.utils.helpers import read_api_endpoint
from taxipred.utils.constants import ML_MODEL_PATH, default_features
from taxipred.training.visualize import plot_feature_importance
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

all_data = read_api_endpoint("all-taxi-rides")
prediction = read_api_endpoint("taxi-price-prediction")

all_data_df = pd.DataFrame(all_data.json())

model = joblib.load(ML_MODEL_PATH)

def main():
    st.markdown("# Taxi Prediction Dashboard")

    tab1, tab2, tab3 = st.tabs(["Show all data", "Make your own prediction", "Feature importance"])

    with tab1:
        st.markdown("### All data tab")
        if st.button("Show all data"):
            st.dataframe(all_data_df)

    with tab2: 
        trip_distance_km = st.number_input("Trip distance in km: ", min_value=1, max_value=200, value=10)        
        per_km_rate = st.number_input("Per km (0.5 - 2$): ", min_value=0.5, max_value=2.0, value=1.0)
        per_minute_rate = st.number_input("Per minute (0.1 - 1$): ", min_value=0.1, max_value=1.0, value=0.1)        
        trip_duration_minutes = st.number_input("Trip in minutes: ", min_value=5, max_value=200, value=15)

        features_to_model = default_features.copy()

        features_to_model.update({
            "Trip_Distance_km": trip_distance_km,
            "Per_Km_Rate": per_km_rate,
            "Per_Minute_Rate": per_minute_rate,
            "Trip_Duration_Minutes": trip_duration_minutes
        })

        if st.button("Make prediction: "):
            X = pd.DataFrame([features_to_model])
            y_pred = model.predict(X)
            st.write(f"{y_pred[0].round(1)}$ for this trip")

    with tab3:
        st.markdown("### Feature importance")
        if st.button("ℹ️ What is this?"):
            st.write("Feature importance shows which factors that the model thinks are most important for its price prediction")            
        fig = plot_feature_importance(model, list(features_to_model.keys()))
        st.pyplot(fig)




if __name__ == "__main__":
    main()
