from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import tensorflow as tf
import os
from weather_api import get_city_coordinates, get_daily_forecasts
from prediction_service import predict_weather   


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "feature_extractor.h5")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)

with open(XGB_MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# === FastAPI App ===
app = FastAPI(title="Weather Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🌤 Weather Prediction API is running!"}

@app.get("/weather/predict/{city}")
def get_prediction(city: str):
    lat, lon = get_city_coordinates(city)
    if not lat or not lon:
        return {"error": "City not found"}

    forecast_data = get_daily_forecasts(lat, lon)
    if not forecast_data:
        return {"error": "Could not fetch forecast"}

    predictions = predict_weather(forecast_data, feature_extractor, xgb_model, label_encoder)
    return {"city": city.title(), "predictions": predictions}
