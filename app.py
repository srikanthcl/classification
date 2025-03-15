import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI()

# Define request body format
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([np.array(data.features)])
    return {"prediction": int(prediction[0])}
