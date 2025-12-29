# System Architecture

## Overview

The Student Performance Risk Prediction System follows a **modular, three-tier architecture**:

1. **Data/ML Layer** - Data processing and model training
2. **API Layer** - REST API for serving predictions
3. **Presentation Layer** - Web interface for users

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│                   (Streamlit Frontend)                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────┐
│                       API Layer                          │
│                    (FastAPI Backend)                     │
└────────────────────┬────────────────────────────────────┘
                     │ Load Models
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    Data/ML Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐ │
│  │  Data    │  │  Prep    │  │ Models │  │   Eval   │ │
│  │  Loader  │→ │  rocessing│→│Training│→ │  uation  │ │
│  └──────────┘  └──────────┘  └────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Data/ML Layer (`src/`)

#### `data_loader.py`
- **Purpose**: Load CSV data into pandas DataFrame
- **Key Function**: `load_data(filepath)`
- **Validation**: Checks file existence, handles errors
- **Output**: pandas DataFrame

#### `preprocessing.py`
- **Purpose**: Transform raw data for ML models
- **Key Class**: `Preprocessor`
- **Features**:
  - Missing value imputation (median strategy)
  - Feature scaling (StandardScaler)
  - State persistence (fitted scaler saved)
- **Methods**:
  - `fit_transform()` - Fit on training data and transform
  - `transform()` - Transform new data using fitted parameters
  - `split_data()` - Train/test split

**Why this design?**
- Separates fitting (training) from transforming (inference)
- Ensures consistent preprocessing between training and deployment
- Prevents data leakage

#### `models.py`
- **Purpose**: Define and train ML models
- **Key Classes**:
  - `ModelFactory` - Creates model instances
  - `ModelTrainer` - Trains and manages models
- **Supported Models**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
- **Features**:
  - Unified interface for all models
  - Model persistence with joblib
  - Probability prediction support

#### `evaluation.py`
- **Purpose**: Evaluate model performance
- **Key Class**: `Evaluator`
- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Feature Importance**: Extracts importance from tree-based models or coefficients from linear models

### 2. API Layer (`src/app.py`)

#### FastAPI Application

**Endpoints:**

```python
@app.get("/health")
# Health check and model status

@app.post("/predict")
# Single student prediction
# Input: JSON with student features
# Output: {at_risk: 0/1, risk_probability: float}

@app.post("/predict_batch")
# Batch prediction from CSV
# Input: File upload (multipart/form-data)
# Output: JSON array with predictions
```

**Startup Process:**
1. Import custom preprocessing class
2. Load saved model (`RandomForest.pkl`)
3. Load saved preprocessor (`preprocessor.pkl`)
4. Validate artifacts loaded successfully

**Request Flow:**
```
Client Request
    ↓
FastAPI receives JSON/File
    ↓
Validate with Pydantic
    ↓
Convert to DataFrame
    ↓
preprocessor.transform()
    ↓
model.predict()
    ↓
Return JSON response
```

**Error Handling:**
- 503: Model not loaded
- 500: Processing error
- Detailed error messages for debugging

### 3. Presentation Layer (`src/frontend.py`)

#### Streamlit Web Application

**Features:**
- **Health Check**: Validates backend connection
- **Single Prediction Mode**:
  - Interactive sliders/inputs
  - Real-time API calls
  - Visual risk display with progress bars
- **Batch Analysis Mode**:
  - CSV upload
  - Summary statistics (total, at-risk, safe)
  - Downloadable results
  - Color-coded risk highlighting

**Architecture Pattern:**
```
Streamlit UI
    ↓
HTTP Request to FastAPI
    ↓
Display Results
```

**Why Streamlit?**
- Rapid development
- Built-in widgets (sliders, file upload)
- No frontend framework knowledge needed
- Great for data applications

## Training Pipeline (`main.py`)

### Workflow

```
1. Load Data
    ↓
2. Preprocess
   - Fit imputer and scaler
   - Transform data
    ↓
3. Train/Test Split (80/20)
    ↓
4. Train Multiple Models
   - 5-Fold Cross-Validation
   - Evaluate on test set
    ↓
5. Select Best Model
   - Based on F1-Score
    ↓
6. Save Artifacts
   - model.pkl
   - preprocessor.pkl
    ↓
7. Generate Report
   - Comparison table
   - Feature importance
    ↓
8. Create Predictions CSV
```

### Cross-Validation Strategy

We use **StratifiedKFold** with 5 folds:
- Preserves class distribution in each fold
- Reduces overfitting risk
- Provides robust performance estimate

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
```

## Data Flow

### Training Phase
```
CSV File
  ↓
Load (data_loader.py)
  ↓
DataFrame
  ↓
Preprocess (preprocessing.py)
  ↓ fit_transform()
Scaled Features + Target
  ↓
Split (80/20)
  ↓
Train Models (models.py)
  ↓ cross_val_score()
Select Best
  ↓
Save (joblib)
  ↓
RandomForest.pkl + preprocessor.pkl
```

### Inference Phase
```
New Student Data (JSON/CSV)
  ↓
API Endpoint (app.py)
  ↓
Convert to DataFrame
  ↓
preprocessor.transform()
  ↓
model.predict()
  ↓
Risk Prediction + Probability
  ↓
JSON Response
  ↓
Frontend Display
```

## Design Patterns Used

### 1. Factory Pattern
`ModelFactory.get_model()` creates model instances based on string identifier

### 2. Strategy Pattern
Different preprocessing strategies encapsulated in `Preprocessor` class

### 3. Facade Pattern
`main.py` provides simple interface to complex ML pipeline

### 4. Stateful Preprocessing
Preprocessor maintains fitted state (scaler parameters) for consistent transformation

## Technology Choices

| Component | Technology | Reasoning |
|-----------|-----------|-----------|
| ML Framework | scikit-learn | Industry standard, comprehensive, well-documented |
| API Framework | FastAPI | Modern, fast, automatic docs, type validation |
| Web Server | Uvicorn | ASGI server, async support, production-ready |
| Frontend | Streamlit | Rapid development, data-focused, Python-native |
| Serialization | joblib | Efficient for large numpy arrays, scikit-learn standard |
| Data Processing | pandas | De facto standard for tabular data in Python |

## Scalability Considerations

### Current Design (Single Machine)
- Suitable for: 100-1000 predictions/day
- Bottleneck: Single FastAPI instance

### Future Scaling Options

**Horizontal Scaling:**
```
Load Balancer
    ↓
FastAPI Instance 1, 2, 3, ...
    ↓
Shared Model Storage (S3/Azure Blob)
```

**Batch Processing:**
- Use Celery for async batch jobs
- Queue large CSV files
- Email results when complete

**Database Integration:**
- Store predictions in PostgreSQL/MongoDB
- Enable historical analysis
- Support model retraining pipeline

## Security Considerations

**Current State:**
- No authentication (development only)
- Local network only

**Production Requirements:**
- API key authentication
- HTTPS (TLS/SSL)
- Rate limiting
- Input validation (already using Pydantic)
- CORS configuration

## File Dependencies

```
main.py depends on:
  ├── src/data_loader.py
  ├── src/preprocessing.py
  ├── src/models.py
  └── src/evaluation.py

src/app.py depends on:
  ├── src/preprocessing.py (for unpickling)
  ├── models/RandomForest.pkl
  └── models/preprocessor.pkl

src/frontend.py depends on:
  └── src/app.py (via HTTP)
```

## Configuration Management

**Current:** Hardcoded paths and parameters

**Recommended for Production:**
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/RandomForest.pkl"
    PREPROCESSOR_PATH: str = "models/preprocessor.pkl"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
```

## Monitoring & Logging

**Current:** Print statements

**Production Recommendations:**
- Use Python `logging` module
- Log prediction requests and responses
- Monitor API latency
- Track model performance drift
- Set up alerts for errors

## Testing Strategy

**Unit Tests:**
- Test each module independently
- Mock external dependencies

**Integration Tests:**
- Test API endpoints
- Test full pipeline

**Current Test:**
`test_api.py` - Basic integration test for prediction endpoint
