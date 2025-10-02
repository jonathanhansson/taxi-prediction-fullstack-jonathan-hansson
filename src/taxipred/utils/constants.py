import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TAXI_CSV_PATH = os.path.join(BASE_DIR, "data", "taxi_trip_pricing.csv")
ML_MODEL_PATH = os.path.join(BASE_DIR, "backend", "taxi_price_model.pkl")

# DATA_PATH = Path(__file__).parents[1] / "data"

"""
I will import this in the frontend
so that the end user wont have to
input ALL the features, only a couple.
"""
default_features = {
    "Trip_Distance_km": 10,
    "Passenger_Count": 2,
    "Base_Fare": 2,
    "Per_Km_Rate": 1.0,
    "Per_Minute_Rate": 0.1,
    "Trip_Duration_Minutes": 15,
    "Traffic_Conditions_High": 0,
    "Traffic_Conditions_Low": 0,
    "Traffic_Conditions_Medium": 1,
    "Weather_Clear": 1,
    "Weather_Rain": 0,
    "Weather_Snow": 0,
    "Time_of_Day_Afternoon": 0,
    "Time_of_Day_Evening": 0,
    "Time_of_Day_Morning": 1,
    "Time_of_Day_Night": 0,
    "Day_of_Week_Weekday": 1,
    "Day_of_Week_Weekend": 0
}

if __name__ == '__main__':
    print(BASE_DIR)
    print(TAXI_CSV_PATH)
    print(ML_MODEL_PATH)
    