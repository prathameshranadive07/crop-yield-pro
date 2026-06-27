import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



print("🚀 Training Multiple Models...")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_production.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

def train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # Feature Engineering
    df['Yield'] = df['Production'] / df['Area']

    # Remove outliers (important)
    df = df[df['Yield'] < df['Yield'].quantile(0.99)]

    # Features
    X = df[['State_Name', 'District_Name', 'Crop', 'Season', 'Area']]
    y = df['Yield']

    categorical = ['State_Name', 'District_Name', 'Crop', 'Season']
    numerical = ['Area']

    preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical),
    ('num', 'passthrough', numerical)
])

    # MODELS DICTIONARY
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=10),
        "Random Forest": RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
),
        "XGBoost": XGBRegressor(n_estimators=100, n_jobs=-1)
    }

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    best_score = -999
    best_model = None

    print("\n📊 MODEL COMPARISON:\n")

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        print(f"⏳ Training {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")

        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name = name

    print("\n🏆 Best Model:", best_name)
    print("Best R2 Score:", best_score)

    # Save best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("💾 Best model saved!")

if __name__ == "__main__":
    train()

    