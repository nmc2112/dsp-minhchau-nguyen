import joblib
import numpy as np
import pandas as pd


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    model = joblib.load('../models/model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    encoder = joblib.load('../models/encoder.joblib')

    selected_continuous = ['GrLivArea', 'LotArea']
    selected_categorical = ['Neighborhood', 'HouseStyle']
    X_cont_scaled = scaler.transform(input_data[selected_continuous])
    X_cat_encoded = encoder.transform(input_data[selected_categorical])
    X_infer = np.hstack((X_cont_scaled, X_cat_encoded))
    predictions = model.predict(X_infer)

    return predictions
