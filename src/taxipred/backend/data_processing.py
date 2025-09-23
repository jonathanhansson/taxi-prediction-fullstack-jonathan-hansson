from taxipred.utils.constants import TAXI_CSV_PATH
import pandas as pd
import json


class TaxiData:
    def __init__(self):
        self.df = pd.read_csv(TAXI_CSV_PATH)

    def preprocess_data(self):
        categorical_cols = [
            'Traffic_Conditions', 
            'Weather', 
            'Time_of_Day', 
            'Day_of_Week'
        ]

        self.df = pd.get_dummies(self.df, columns=categorical_cols)

        return self.df.dropna()
    
    def get_features_and_target(self, target='Trip_Price'):
        df_processed = self.preprocess_data()
        X, y = df_processed.drop(target, axis=1), df_processed[target]
        
        return X, y

    def to_json(self):
        return json.loads(self.df.to_json(orient = "records"))
