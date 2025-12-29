# API Reference

Complete reference for all API endpoints.

## Base URL

```
http://127.0.0.1:8000
```

## Authentication

Currently no authentication (development only).

For production, add API key header:
```
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints

### GET `/health`

Health check endpoint to verify API status and model availability.

#### Request

No parameters required.

#### Response

**Status Code:** 200 OK

**Body:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Fields:**
- `status` (string): Always "ok" if API is running
- `model_loaded` (boolean): Whether ML model is successfully loaded

#### Example

**cURL:**
```bash
curl http://127.0.0.1:8000/health
```

**Python:**
```python
import requests
response = requests.get("http://127.0.0.1:8000/health")
print(response.json())
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### POST `/predict`

Predict academic risk for a single student.

#### Request

**Content-Type:** `application/json`

**Body Schema:**
```json
{
  "attendance_percentage": float,
  "assignment_average": float,
  "internal_marks": float,
  "previous_sem_gpa": float
}
```

**Field Constraints:**
- `attendance_percentage`: 0.0 - 100.0
- `assignment_average`: 0.0 - 100.0
- `internal_marks`: 0.0 - 100.0
- `previous_sem_gpa`: 0.0 - 10.0

#### Response

**Status Code:** 200 OK

**Body:**
```json
{
  "at_risk": int,
  "risk_probability": float
}
```

**Fields:**
- `at_risk`: 0 (not at risk) or 1 (at risk)
- `risk_probability`: Probability of being at risk (0.0 - 1.0)

#### Error Responses

**503 Service Unavailable**
```json
{
  "detail": "Model not loaded. Please ensure main.py has been run."
}
```

**500 Internal Server Error**
```json
{
  "detail": "Processing failed: [error details]"
}
```

**422 Unprocessable Entity** (Invalid input)
```json
{
  "detail": [
    {
      "loc": ["body", "attendance_percentage"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### Examples

**cURL:**
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

**Python (requests):**
```python
import requests

payload = {
    "attendance_percentage": 85,
    "assignment_average": 70,
    "internal_marks": 65,
    "previous_sem_gpa": 7.5
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print(response.json())
# Output: {"at_risk": 0, "risk_probability": 0.12}
```

**JavaScript (fetch):**
```javascript
fetch('http://127.0.0.1:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    attendance_percentage: 85,
    assignment_average: 70,
    internal_marks: 65,
    previous_sem_gpa: 7.5
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### POST `/predict_batch`

Predict academic risk for multiple students from CSV file.

#### Request

**Content-Type:** `multipart/form-data`

**Form Field:**
- `file`: CSV file

**CSV Format:**
Must contain columns:
- `attendance_percentage`
- `assignment_average`
- `internal_marks`
- `previous_sem_gpa`

Optional columns (will be preserved in output):
- `student_id`
- Any other columns

**Example CSV:**
```csv
student_id,attendance_percentage,assignment_average,internal_marks,previous_sem_gpa
S1000,93,79,78,7.67
S1001,83,61,66,4.49
S1002,69,66,66,4.03
```

#### Response

**Status Code:** 200 OK

**Body:** Array of JSON objects

```json
[
  {
    "student_id": "S1000",
    "attendance_percentage": 93,
    "assignment_average": 79,
    "internal_marks": 78,
    "previous_sem_gpa": 7.67,
    "Predicted_Risk": 0,
    "Risk_Probability": 0.01
  },
  {
    "student_id": "S1001",
    "attendance_percentage": 83,
    "assignment_average": 61,
    "internal_marks": 66,
    "previous_sem_gpa": 4.49,
    "Predicted_Risk": 1,
    "Risk_Probability": 0.98
  }
]
```

**Added Fields:**
- `Predicted_Risk`: 0 or 1
- `Risk_Probability`: 0.0 - 1.0

#### Error Responses

**503 Service Unavailable**
```json
{
  "detail": "Model not loaded"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Processing failed: Missing required column 'attendance_percentage'"
}
```

#### Examples

**cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict_batch" \
  -F "file=@students.csv"
```

**Python (requests):**
```python
import requests

with open('students.csv', 'rb') as f:
    response = requests.post(
        "http://127.0.0.1:8000/predict_batch",
        files={"file": f}
    )

results = response.json()
print(f"Processed {len(results)} students")

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(results)
print(df[['student_id', 'Predicted_Risk', 'Risk_Probability']])
```

**Python (httpx - async):**
```python
import httpx
import asyncio

async def batch_predict():
    async with httpx.AsyncClient() as client:
        with open('students.csv', 'rb') as f:
            response = await client.post(
                "http://127.0.0.1:8000/predict_batch",
                files={"file": f}
            )
    return response.json()

results = asyncio.run(batch_predict())
```

---

## Interactive Documentation

FastAPI automatically generates interactive docs:

### Swagger UI
Visit http://127.0.0.1:8000/docs

Features:
- Try endpoints directly in browser
- See request/response schemas
- Download OpenAPI spec

### ReDoc
Visit http://127.0.0.1:8000/redoc

Features:
- Clean, readable documentation
- Code samples in multiple languages
- Search functionality

---

## Rate Limiting

Currently no rate limiting.

For production, implement using:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/hour")
def predict(request: Request, data: StudentData):
    # ...
```

---

## CORS Configuration

For production with separate frontend:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Error Codes Summary

| Code | Meaning | Common Cause |
|------|---------|--------------|
| 200 | Success | Request processed successfully |
| 422 | Validation Error | Invalid input format/types |
| 500 | Server Error | Model prediction failed, processing error |
| 503 | Service Unavailable | Model not loaded |

---

## Response Times

Typical response times (local machine):

- `/health`: < 10ms
- `/predict`: 20-50ms
- `/predict_batch` (100 students): 100-200ms

Factors affecting performance:
- Model complexity (RandomForest is fast)
- Batch size
- Server resources

---

## Model Information

Current model details:

**Algorithm:** Random Forest Classifier

**Features (in order):**
1. attendance_percentage
2. assignment_average
3. internal_marks
4. previous_sem_gpa

**Output:**
- Binary classification (0: safe, 1: at-risk)
- Probability scores (0.0 to 1.0)

**Preprocessing:**
- Median imputation for missing values
- StandardScaler normalization

---

## Version Information

To get model version and metadata:

```python
import joblib

model = joblib.load('models/RandomForest.pkl')
print(f"Model type: {type(model).__name__}")
print(f"Number of trees: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
```

---

## Best Practices

### 1. Batch Over Single Predictions
For > 10 students, use `/predict_batch` instead of multiple `/predict` calls.

### 2. Handle Errors Gracefully
```python
try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.ConnectionError:
    print("Cannot connect to API")
```

### 3. Validate Input Locally
Check data ranges before sending to API to avoid 422 errors.

### 4. Cache Results
If predicting same student multiple times, cache the result.

### 5. Use Connection Pooling
```python
import requests

session = requests.Session()
# Reuse for multiple requests
response = session.post(url, json=data)
```
