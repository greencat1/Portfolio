# app/scripts/predict.py
import pandas as pd
from app.scripts.model import load_model
from app.core.database import get_db
from datetime import datetime
from app.utils.logger import logger

def make_prediction(data):
    """Make prediction and save to SQLite database"""
    model = load_model()
    
    # Convert to DataFrame (transformer expects pandas)
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Prepare data for database
    input_dict = data.dict()
    
    # Save to SQLite
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if customer already exists
        existing = cursor.execute(
            "SELECT customerID FROM new_data WHERE customerID = ?",
            (input_dict['customerID'],)
        ).fetchone()
        
        if existing:
            # Update existing record
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
                int(prediction), float(probability), input_dict['customerID']
            ))
            logger.info(f"Updated existing record for customerID: {input_dict['customerID']}")
        else:
            # Insert new record
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
                int(prediction), float(probability)
            ))
            logger.info(f"Added new record for customerID: {input_dict['customerID']}")
        
        conn.commit()
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }