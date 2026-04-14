import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List
from app.scripts.model import load_model
from app.schemas import PredictRequest


# ====================== BATCH PREDICTION ======================
def predict_batch(data_list: List[PredictRequest]):
    """
    Batch prediction function.
    - Makes fast batch prediction
    - Saves each record to CSV (updates if customerID already exists)
    """
    if not data_list:
        return {"predictions": [], "total": 0}

    model = load_model()
    
    # Convert list of PredictRequest to DataFrame for batch inference
    input_list = [item.dict() for item in data_list]
    input_df = pd.DataFrame(input_list)
    
    # Batch prediction (efficient)
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]

    # Save each prediction to CSV
    for i, item in enumerate(data_list):
        _save_single_record(item, predictions[i], probabilities[i])

    # Prepare response
    results = [
        {
            "prediction": int(pred),
            "probability": float(prob)
        }
        for pred, prob in zip(predictions, probabilities)
    ]

    return {
        "predictions": results,
        "total": len(results)
    }


# ====================== INTERNAL SAVE FUNCTION ======================
def _save_single_record(data, prediction: int, probability: float):
    """
    Saves or updates a single record in CSV by customerID
    """
    file_path = Path('app/data/new_data/new_data.csv')
    
    # Prepare record with extra columns
    input_dict = data.dict() if hasattr(data, 'dict') else dict(data)
    input_dict.update({
        'prediction': int(prediction),
        'probability': float(probability),
        'Churn': np.nan,                    # true label is unknown for now
        'timestamp': datetime.utcnow().isoformat()
    })
    
    new_row = pd.DataFrame([input_dict])

    if not file_path.exists():
        # First time - create file
        new_row.to_csv(file_path, index=False)
        print(f"File created. Added customerID: {input_dict.get('customerID')}")
    else:
        # Read existing file
        df = pd.read_csv(file_path)
        
        customer_id_str = str(input_dict.get('customerID'))
        
        if customer_id_str in df['customerID'].astype(str).values:
            # Update existing record
            df = df[df['customerID'].astype(str) != customer_id_str]
            print(f"Updated existing record for customerID: {customer_id_str}")
        else:
            print(f"Added new record for customerID: {customer_id_str}")
        
        # Append new record
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_path, index=False)