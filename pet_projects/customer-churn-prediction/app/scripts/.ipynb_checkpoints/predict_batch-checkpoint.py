# app/predict.py
import pandas as pd
from app.model import load_model

def make_prediction(data):
    model = load_model()
    
    input_df = pd.DataFrame([data.dict()])
    

    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }