import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def feature_engineering(df):
    df = df.dropna()

    # Yield calculation
    df['Yield'] = df['Production'] / df['Area']

    return df

def split_features(df):
    X = df[['State_Name', 'District_Name', 'Crop', 'Season', 'Area']]
    y = df['Yield']
    return X, y