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

@app.get("/taxi-price-prediction")
async def predict(features: dict):
    X = pd.DataFrame()
    y_pred = model.predict(X)
    return {"prediction": float(y_pred[0])}
