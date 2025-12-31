# Getting Started Guide

## Introduction

Welcome to the Student Performance Risk Prediction System! This guide will help you get up and running quickly.

## Prerequisites

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** (optional) - For version control
- **Code Editor** - VS Code, PyCharm, or any editor of your choice
- **Terminal** - PowerShell (Windows), Terminal (Mac/Linux)

## Installation Steps

### Step 1: Navigate to Project Directory

```powershell
cd "d:\Projects\Student Prediction"
```

### Step 2: Create Virtual Environment

A virtual environment isolates project dependencies from your system Python.

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate  # Windows PowerShell
# OR
# source venv/bin/activate  # Mac/Linux
```

You should see `(venv)` prefix in your terminal prompt.

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs all required packages:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `fastapi` - API framework
- `uvicorn` - Web server
- `streamlit` - UI framework
- And more...

### Step 4: Verify Installation

```powershell
python --version
pip list
```

## First Run: Training the Model

Before using the API or UI, you must train the model:

```powershell
python main.py
```

**What this does:**
1. Loads `Dataset/student_performance_risk_dataset.csv`
2. Preprocesses the data (scaling, imputation)
3. Trains 4 different ML models
4. Evaluates each with cross-validation
5. Saves the best model to `models/RandomForest.pkl`
6. Saves the preprocessor to `models/preprocessor.pkl`
7. Generates a report in `reports/model_comparison.md`
8. Creates predictions CSV

**Expected Output:**
```
Loading data...
Successfully loaded data from ... Shape: (120, 6)
Preprocessing data...

Training models (using 5-Fold Cross-Validation)...
Trained LogisticRegression
  LogisticRegression: Test F1=0.9231, CV F1=0.8957
...
Best Model: RandomForest (F1: 1.0000)
Saved RandomForest to ...
```

## Running the Application

### Option A: Web Interface (Recommended for End Users)

**Terminal 1 - Start Backend:**
```powershell
.\venv\Scripts\activate
uvicorn src.app:app --reload
```

Keep this terminal running.

**Terminal 2 - Start Frontend:**
```powershell
.\venv\Scripts\activate
streamlit run src/frontend.py
```

Your browser will open automatically to http://localhost:8501

### Option B: API Only (For Developers)

```powershell
.\venv\Scripts\activate
uvicorn src.app:app --reload
```

Access API docs at: http://127.0.0.1:8000/docs

## Making Your First Prediction

### Via Web UI:
1. Open http://localhost:8501
2. Select "Single Student Prediction"
3. Adjust the sliders:
   - Attendance: 85%
   - Assignment Average: 70
   - Internal Marks: 65
   - Previous Semester GPA: 7.5
4. Click "Predict Risk Status"
5. View the result!

### Via API (curl):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "attendance_percentage": 85,
    "assignment_average": 70,
    "internal_marks": 65,
    "previous_sem_gpa": 7.5
  }'
```

### Via Python (requests):
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "attendance_percentage": 85,
        "assignment_average": 70,
        "internal_marks": 65,
        "previous_sem_gpa": 7.5
    }
)

print(response.json())
# Output: {"at_risk": 0, "risk_probability": 0.12}
```

## Batch Predictions

### Via Web UI:
1. Select "Batch Analysis (CSV)"
2. Upload your CSV file
3. Click "Analyze Batch"
4. Download the results

### Via API:
```python
import requests

with open('student_data.csv', 'rb') as f:
    response = requests.post(
        "http://127.0.0.1:8000/predict_batch",
        files={"file": f}
    )

results = response.json()
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Ensure virtual environment is activated and dependencies installed
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Model not loaded" error
**Solution:** Train the model first
```powershell
python main.py
```

### Issue: Port 8000 already in use
**Solution:** Kill existing process or use different port
```powershell
# Use different port
uvicorn src.app:app --port 8001
```

### Issue: Streamlit won't open
**Solution:** Manually open http://localhost:8501 in browser

## Next Steps

- Read [Architecture Overview](architecture.md) to understand system design
- Check [Code Walkthrough](code_walkthrough.md) for detailed code explanations
- See [API Reference](api_reference.md) for complete endpoint documentation
- Review [Development Guide](development_guide.md) for contributing

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python main.py` | Train models |
| `uvicorn src.app:app --reload` | Start API server |
| `streamlit run src/frontend.py` | Start web UI |
| `python test_api.py` | Test API endpoints |
| `.\venv\Scripts\activate` | Activate virtual env |
| `deactivate` | Deactivate virtual env |

## Support

For issues or questions, refer to:
- [Troubleshooting Guide](troubleshooting.md)
- Check the code comments in source files
- Review the Jupyter notebook in `notebooks/`
