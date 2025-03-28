import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_continuous_features(data: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(data[selected_columns])
    return scaler.transform(data[selected_columns]), scaler

def preprocess_categorical_features(data: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(data[selected_columns])
    return encoder.transform(data[selected_columns]), encoder