
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import io
import sys

# Add src to path if needed for pickle loading custom classes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import Preprocessor 

app = FastAPI(title="Student Risk Prediction API")

MODEL_PATH = "models/RandomForest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

# Load artifacts
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Model and Preprocessor loaded successfully.")
    else:
        model = None
        preprocessor = None
        print("Artifacts not found.")
except Exception as e:
    model = None
    preprocessor = None
    print(f"Error loading artifacts: {e}")

class StudentData(BaseModel):
    attendance_percentage: float
    assignment_average: float
    internal_marks: float
    previous_sem_gpa: float

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: StudentData):
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure main.py has been run.")
    
    # Create DataFrame from input
    df = pd.DataFrame([data.dict()])
    
    try:
        # Transform
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1] if hasattr(model, "predict_proba") else None
        
        return {
            "at_risk": int(prediction),
            "risk_probability": float(probability) if probability is not None else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if not model or not preprocessor:
         raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Transform
        X_processed = preprocessor.transform(df)
        
        # Predict
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1] if hasattr(model, "predict_proba") else [0.0]*len(predictions)
        
        # Prepare Result
        results = df.copy()
        results['Predicted_Risk'] = predictions
        results['Risk_Probability'] = probabilities
        
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
