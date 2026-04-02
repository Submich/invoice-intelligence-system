import pandas as pd
import joblib

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_scaler(scaler_path: str = SCALER_PATH):
    with open(scaler_path, "rb") as file:
        scaler = joblib.load(file)
    return scaler


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_invoice_flag(input_data: dict):

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    features = [
        'invoice_quantity',
        'invoice_dollars',
        'Freight',
        'total_item_quantity',
        'total_item_dollars'
    ]

    #  Validate input
    missing_cols = set(features) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    #  Enforce correct order
    input_df = input_df[features]

    #  Apply scaling
    input_scaled = scaler.transform(input_df)

    # Predict
    input_df['Predicted_Flag'] = model.predict(input_scaled)

    return input_df


#  Entry point for testing
if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [5600, 1350],
        "invoice_dollars": [213, 3300],
        "Freight": [2000, 1233],
        "total_item_quantity": [12, 275],
        "total_item_dollars": [1890, 8000]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)