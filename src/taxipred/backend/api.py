import pandas as pd
from fastapi import FastAPI
from taxipred.backend.data_processing import TaxiData
import joblib

app = FastAPI()

# instantiating a TaxiData object
taxi_data = TaxiData()

# loading the pretrained model
model = joblib.load("src/taxipred/backend/taxi_price_model.pkl")


@app.get("/taxi/")
async def read_taxi_data():
    return taxi_data.to_json()

@app.get("/taxi/prediction")
async def predict(features: dict):
    X = pd.DataFrame()
    y_pred = model.predict(X)
    return {"prediction": float(y_pred[0])}
