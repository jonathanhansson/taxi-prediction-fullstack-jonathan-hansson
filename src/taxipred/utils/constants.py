import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TAXI_CSV_PATH = os.path.join(BASE_DIR, "data", "taxi_trip_pricing.csv")
ML_MODEL_PATH = os.path.join(BASE_DIR, "backend", "taxi_price_model.pkl")

# DATA_PATH = Path(__file__).parents[1] / "data"

if __name__ == '__main__':
    print(BASE_DIR)
    print(TAXI_CSV_PATH)
    print(ML_MODEL_PATH)
    