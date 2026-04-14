# app/predict.py
import pandas as pd
from app.scripts.model import load_model
import numpy as np
from pathlib import Path
from datetime import datetime

def make_prediction(data):
    model = load_model()
    
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    file_path = Path('app/data/new_data/new_data.csv')
    
    # Prepare the new record with additional columns
    input_dict = data.dict()
    input_dict.update({
        'prediction': int(prediction),
        'probability': float(probability),
        'Churn': np.nan,                    # true label is unknown for now
        'timestamp': datetime.utcnow().isoformat()
    })
    
    new_row = pd.DataFrame([input_dict])

    if not file_path.exists():
        # First time: create new file with header
        new_row.to_csv(file_path, index=False)
        print(f"File created. Added customerID: {data.customerID}")
        
    else:
        # Read existing data
        df = pd.read_csv(file_path)
        
        # Check if customerID already exists
        customer_id_str = str(data.customerID)
        
        if customer_id_str in df['customerID'].astype(str).values:
            # Remove old record for this customer
            df = df[df['customerID'].astype(str) != customer_id_str]
            print(f"Updated existing record for customerID: {data.customerID}")
        else:
            print(f"Added new record for customerID: {data.customerID}")
        
        # Append the new record
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(file_path, index=False)
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }