import joblib
import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "models", "random_forest_extend.joblib")
full_model = joblib.load(model_path)
model = full_model["model"]
scaler = full_model["scaler"]
features = full_model["features"]
scale_cols = full_model["scale_cols"]

def call_func(brand, seller_type, transmission, year, owner, fuel_type, engine, max_power):
    input_df = pd.DataFrame({
        'transmission': [transmission],
        'seller_type': [seller_type],
        'year': [2020 - year], 
        'owner': [owner],
        'fuel': [fuel_type],
        'max_power': [max_power],
        'engine': [engine]
    })
    
    input_df[brand] = 1

    input_df = input_df.reindex(columns=features, fill_value=0)
    
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    log_price = model.predict(input_df)
    return np.exp(log_price)[0]