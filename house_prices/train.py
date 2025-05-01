import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn
from house_prices.preprocess import (
    preprocess_continuous_features, preprocess_categorical_features
)

def build_model(data: pd.DataFrame) -> dict:
    selected_continuous = ['GrLivArea', 'LotArea']
    selected_categorical = ['Neighborhood', 'HouseStyle']

    X_cont_scaled, scaler = preprocess_continuous_features(data, selected_continuous)
    X_cat_encoded = preprocess_categorical_features(data, selected_categorical)

    X = np.hstack((X_cont_scaled, X_cat_encoded))
    y = data['SalePrice']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    from sklearn.metrics import mean_squared_log_error

    def compute_rmsle(y_true, y_pred, precision=2):
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
        return round(rmsle, precision)

    y_pred = model.predict(X_val)
    rmsle_error = compute_rmsle(y_val, y_pred)

    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(encoder, 'models/encoder.joblib')

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("selected_continuous", selected_continuous)
        mlflow.log_param("selected_categorical", selected_categorical)
        mlflow.log_metric("val_rmsle", rmsle_error)

        mlflow.sklearn.log_model(model, "model")

    return {'rmsle': rmsle_error}

def main():
    # Load your dataset
    data = pd.read_csv('data/house_prices.csv')  # make sure you have the data here

    # Build and train the model
    result = build_model(data)

    # Print result
    print(f"Training complete. Validation RMSLE: {result['rmsle']}")

if __name__ == "__main__":
    main()
