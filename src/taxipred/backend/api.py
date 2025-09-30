import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from taxipred.utils.constants import ML_MODEL_PATH 
from taxipred.backend.data_processing import TaxiData
import joblib

app = FastAPI()
taxi_data = TaxiData()

# loading the pretrained model
model = joblib.load(ML_MODEL_PATH)

class TaxiFeatures(BaseModel):
    features: dict

@app.get("/all-taxi-rides")
async def read_taxi_data():
    return taxi_data.to_json()

"""
Because we will be receving features (data) from the
frontend as a JSON-object, we will use app.post() instead
of app.get().
"""
@app.post("/taxi-price-prediction")
async def predict(features_input: TaxiFeatures):
    features_dict = features_input.features
    X = pd.DataFrame([features_dict])
    y_pred = model.predict(X)
    return {"prediction": float(y_pred[0])}
