import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "crop_production.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2