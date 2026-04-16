# app/scripts/predict_batch.py
import pandas as pd
from typing import List
from app.scripts.model import load_model
from app.core.database import get_db
from app.schemas import PredictRequest
from app.utils.logger import logger

def predict_batch(data_list: List[PredictRequest]):
    """Batch prediction with SQLite storage"""
    if not data_list:
        return {"predictions": [], "total": 0}
    
    model = load_model()
    
    # Convert to DataFrame for batch inference
    input_list = [item.dict() for item in data_list]
    input_df = pd.DataFrame(input_list)
    
    # Batch prediction
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]
    
    # Save each to SQLite
    with get_db() as conn:
        cursor = conn.cursor()
        
        for i, item in enumerate(data_list):
            input_dict = item.dict()
            pred = int(predictions[i])
            prob = float(probabilities[i])
            
            existing = cursor.execute(
                "SELECT customerID FROM new_data WHERE customerID = ?",
                (input_dict['customerID'],)
            ).fetchone()
            
            if existing:
                cursor.execute('''
                    UPDATE new_data SET
                        gender = ?, SeniorCitizen = ?, Partner = ?, Dependents = ?,
                        tenure = ?, PhoneService = ?, MultipleLines = ?, InternetService = ?,
                        OnlineSecurity = ?, OnlineBackup = ?, DeviceProtection = ?,
                        TechSupport = ?, StreamingTV = ?, StreamingMovies = ?,
                        Contract = ?, PaperlessBilling = ?, PaymentMethod = ?,
                        MonthlyCharges = ?, TotalCharges = ?,
                        prediction = ?, probability = ?, created_at = CURRENT_TIMESTAMP
                    WHERE customerID = ?
                ''', (
                    input_dict['gender'], input_dict['SeniorCitizen'], input_dict['Partner'],
                    input_dict['Dependents'], input_dict['tenure'], input_dict['PhoneService'],
                    input_dict['MultipleLines'], input_dict['InternetService'],
                    input_dict['OnlineSecurity'], input_dict['OnlineBackup'],
                    input_dict['DeviceProtection'], input_dict['TechSupport'],
                    input_dict['StreamingTV'], input_dict['StreamingMovies'],
                    input_dict['Contract'], input_dict['PaperlessBilling'],
                    input_dict['PaymentMethod'], input_dict['MonthlyCharges'],
                    float(input_dict['TotalCharges']) if input_dict['TotalCharges'] else 0,
                    pred, prob, input_dict['customerID']
                ))
            else:
                cursor.execute('''
                    INSERT INTO new_data (
                        customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
                        PhoneService, MultipleLines, InternetService, OnlineSecurity,
                        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                        MonthlyCharges, TotalCharges, prediction, probability, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    input_dict['customerID'], input_dict['gender'], input_dict['SeniorCitizen'],
                    input_dict['Partner'], input_dict['Dependents'], input_dict['tenure'],
                    input_dict['PhoneService'], input_dict['MultipleLines'],
                    input_dict['InternetService'], input_dict['OnlineSecurity'],
                    input_dict['OnlineBackup'], input_dict['DeviceProtection'],
                    input_dict['TechSupport'], input_dict['StreamingTV'],
                    input_dict['StreamingMovies'], input_dict['Contract'],
                    input_dict['PaperlessBilling'], input_dict['PaymentMethod'],
                    input_dict['MonthlyCharges'],
                    float(input_dict['TotalCharges']) if input_dict['TotalCharges'] else 0,
                    pred, prob
                ))
        
        conn.commit()
    
    # Prepare response
    results = [
        {"prediction": int(pred), "probability": float(prob)}
        for pred, prob in zip(predictions, probabilities)
    ]
    
    return {
        "predictions": results,
        "total": len(results)
    }