from pydantic import BaseModel

class PredictRequest(BaseModel):
    customerID : object 
    gender : object 
    SeniorCitizen : int
    Partner : object 
    Dependents : object 
    tenure : int
    PhoneService : object
    MultipleLines : object
    InternetService : object
    OnlineSecurity : object
    OnlineBackup : object
    DeviceProtection : object
    TechSupport : object
    StreamingTV : object
    StreamingMovies : object
    Contract : object
    PaperlessBilling : object
    PaymentMethod : object
    MonthlyCharges : float
    TotalCharges : object

    class Config:
        
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85"
            }
        }

class PredictResponse(BaseModel):
    prediction: int
    probability: float