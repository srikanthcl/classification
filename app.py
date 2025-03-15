import joblib
import numpy as np
from fastapi import FastAPI

# Load trained model
model = joblib.load("model.pkl")

app = FastAPI()

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([np.array(features)])
    return {"prediction": int(prediction[0])}
