from taxipred.backend.data_processing import TaxiData
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import sys

taxi_data = TaxiData()

# I call for the method in data_processing.py
# It is important to not call for 'preprocess_data' first, though, 
# since "get_features_and_target()" calls for it already
X, y = taxi_data.get_features_and_target()

# I chose gradient boosting regression because it provided the best evaluation scores
# compared to random forest and linear regression
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

joblib.dump(model, "src/taxipred/backend/taxi_price_model.pkl")

