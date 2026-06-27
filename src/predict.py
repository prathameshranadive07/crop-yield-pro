import joblib
import pandas as pd
from config import MODEL_PATH

def predict(data_dict):
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([data_dict])

    prediction = model.predict(df)

    return prediction[0]
