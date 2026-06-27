import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommendation.pkl")

def recommend_crop(data_dict):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)

    return prediction[0]