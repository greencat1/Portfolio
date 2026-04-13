from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.predict import make_prediction

app = FastAPI()

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    result = make_prediction(data)
    return result