# Code Walkthrough

This document provides detailed explanations of each source file and how they work together.

## Core Modules (`src/`)

### `data_loader.py`

**Purpose:** Load student data from CSV files with error handling.

```python
import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Check if file exists before attempting to load
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    try:
        # Load CSV into DataFrame
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        # Catch and re-raise with context
        raise Exception(f"Error loading data: {e}")
```

**Key Points:**
- Simple, single-purpose function
- Defensive programming (check file existence)
- Informative error messages
- Prints confirmation with data shape

**Usage Example:**
```python
from src.data_loader import load_data

df = load_data('student_performance_risk_dataset.csv')
# Output: Successfully loaded data from student_performance_risk_dataset.csv. Shape: (120, 6)
```

---

### `preprocessing.py`

**Purpose:** Transform raw data into ML-ready format with state persistence.

#### Class: `Preprocessor`

**Attributes:**
- `scaler`: StandardScaler instance (scales features to mean=0, std=1)
- `imputer`: SimpleImputer instance (fills missing values with median)
- `feature_names`: List of feature column names (stored after fitting)

**Method: `fit_transform()`**

```python
def fit_transform(self, df: pd.DataFrame, target_column: str = 'at_risk'):
    """
    Fits the imputer and scaler on the data and transforms it.
    
    This should ONLY be used during training.
    """
    # Separate features from target
    X, y = self._get_X_y(df, target_column, training=True)
    
    # Step 1: Fit imputer and fill missing values
    X_imputed = self.imputer.fit_transform(X)
    self.feature_names = X.columns.tolist()
    
    # Step 2: Fit scaler and normalize features
    X_scaled = self.scaler.fit_transform(X_imputed)
    
    # Return as numpy array and separate target
    return X_scaled, y, full_processed_df, self.feature_names
```

**Why fit_transform?**
- Calculates statistics (median, mean, std) from training data
- Applies transformation using those statistics
- Stores statistics in the object for later use

**Method: `transform()`**

```python
def transform(self, df: pd.DataFrame):
    """
    Transforms new data using *already fitted* imputer and scaler.
    
    This is used for inference (new predictions).
    """
    X, _ = self._get_X_y(df, training=False)
    
    # Use previously computed medians
    X_imputed = self.imputer.transform(X)
    
    # Use previously computed mean/std
    X_scaled = self.scaler.transform(X_imputed)
    
    return X_scaled
```

**Critical Difference:**
- `fit_transform()`: Learn from data, then transform
- `transform()`: Only transform using learned parameters

**Example:**
```python
# Training
preprocessor = Preprocessor()
X_train, y_train, _, features = preprocessor.fit_transform(train_df)

# Later, for new data (inference)
X_new = preprocessor.transform(new_student_df)
# Uses the same median and scaling as training!
```

**Why This Matters:**
If you calculated a different mean/std for new data, the model predictions would be meaningless because the scale is different!

---

### `models.py`

**Purpose:** Provide unified interface for multiple ML algorithms.

#### Class: `ModelFactory`

```python
class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        """Factory method to create model instances."""
        if model_name == 'LogisticRegression':
            return LogisticRegression(random_state=42, **kwargs)
        elif model_name == 'RandomForest':
            return RandomForestClassifier(random_state=42, **kwargs)
        elif model_name == 'SVM':
            return SVC(probability=True, random_state=42, **kwargs)
        elif model_name == 'GradientBoosting':
            return GradientBoostingClassifier(random_state=42, **kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported.")
```

**Benefits:**
- Centralized model configuration
- Easy to add new models
- Consistent random_state for reproducibility

#### Class: `ModelTrainer`

```python
class ModelTrainer:
    def __init__(self):
        self.models = {}  # Dictionary to store trained models
        
    def train_model(self, model_name: str, X_train, y_train, **kwargs):
        """Train and store a model."""
        model = ModelFactory.get_model(model_name, **kwargs)
        model.fit(X_train, y_train)
        self.models[model_name] = model  # Store for later use
        return model
    
    def predict(self, model_name: str, X_test):
        """Get predictions from a stored model."""
        return self.models[model_name].predict(X_test)
    
    def save_model(self, model_name: str, filepath: str):
        """Serialize model to disk."""
        joblib.dump(self.models[model_name], filepath)
```

**Usage:**
```python
trainer = ModelTrainer()

# Train multiple models
trainer.train_model('RandomForest', X_train, y_train)
trainer.train_model('LogisticRegression', X_train, y_train)

# Make predictions
rf_predictions = trainer.predict('RandomForest', X_test)
lr_predictions = trainer.predict('LogisticRegression', X_test)

# Save best model
trainer.save_model('RandomForest', 'models/RandomForest.pkl')
```

---

### `evaluation.py`

**Purpose:** Calculate performance metrics and extract feature importance.

#### Class: `Evaluator`

```python
class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        """Calculate classification metrics."""
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred)
        }
```

**Metrics Explained:**
- **Accuracy**: % of correct predictions overall
- **Precision**: Of predicted "at-risk", how many actually are?
- **Recall**: Of actual "at-risk", how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)

**Feature Importance:**
```python
@staticmethod
def get_feature_importance(model, feature_names):
    """Extract importance values from model."""
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (RandomForest, GradientBoosting)
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models (LogisticRegression)
        importances = np.abs(model.coef_[0])
    
    # Create sorted DataFrame
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return feature_imp
```

---

## Main Pipeline (`main.py`)

**Purpose:** Orchestrate the entire training workflow.

### Step-by-Step Breakdown

#### 1. Load Data
```python
DATA_PATH = os.path.join(os.getcwd(), 'student_performance_risk_dataset.csv')
df = load_data(DATA_PATH)
```

#### 2. Preprocess
```python
preprocessor = Preprocessor()
X_scaled, y, full_processed_df, feature_names = preprocessor.fit_transform(df)
```
- Fits imputer and scaler on full dataset
- Returns transformed features and target

#### 3. Train/Test Split
```python
X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
```
- 80% training, 20% testing
- Random but reproducible (random_state=42)

#### 4. Train Multiple Models with Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name in ['LogisticRegression', 'RandomForest', 'SVM', 'GradientBoosting']:
    # Cross-validation on FULL dataset
    model = ModelFactory.get_model(model_name)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
    mean_cv_f1 = cv_scores.mean()
    
    # Train on split for test evaluation
    trainer.train_model(model_name, X_train, y_train)
    y_pred = trainer.predict(model_name, X_test)
    metrics = evaluator.evaluate(y_test, y_pred)
```

**Why Cross-Validation?**
- Single train/test split can be lucky or unlucky
- CV averages over 5 different splits
- More reliable performance estimate

#### 5. Select Best Model
```python
if metrics['F1-Score'] > best_f1:
    best_f1 = metrics['F1-Score']
    best_model_name = model_name
```

#### 6. Save Artifacts
```python
# Save model
trainer.save_model(best_model_name, MODEL_SAVE_PATH)

# Save preprocessor (critical for deployment!)
joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
```

**Why Save Preprocessor?**
New data must be transformed identically to training data. The preprocessor contains the exact median values and scaling parameters used during training.

#### 7. Generate Report
```python
with open(REPORT_PATH, 'w') as f:
    f.write("# Model Comparison Report\n\n")
    f.write("| Model | Accuracy | Precision | Recall | Test F1 | CV F1 |\n")
    # ... write table
```

---

## API Backend (`src/app.py`)

**Purpose:** Serve predictions via HTTP endpoints.

### Startup

```python
# Load artifacts at startup
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
```

**Important:** This happens ONCE when server starts, not on every request!

### Endpoint: `/predict`

```python
class StudentData(BaseModel):
    # Pydantic model for type validation
    attendance_percentage: float
    assignment_average: float
    internal_marks: float
    previous_sem_gpa: float

@app.post("/predict")
def predict(data: StudentData):
    # 1. Convert Pydantic model to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # 2. Preprocess (using saved preprocessor's transform)
    X_processed = preprocessor.transform(df)
    
    # 3. Predict
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]
    
    # 4. Return JSON
    return {
        "at_risk": int(prediction),
        "risk_probability": float(probability)
    }
```

**Data Flow:**
```
JSON Input
↓
Pydantic Validation (type checking)
↓
Convert to DataFrame
↓
Preprocessor transforms (using fitted parameters)
↓
Model predicts
↓
JSON Output
```

### Endpoint: `/predict_batch`

```python
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # 1. Read uploaded file
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # 2. Transform entire DataFrame
    X_processed = preprocessor.transform(df)
    
    # 3. Predict for all rows
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    # 4. Add predictions to original data
    results = df.copy()
    results['Predicted_Risk'] = predictions
    results['Risk_Probability'] = probabilities
    
    return results.to_dict(orient="records")
```

---

## Frontend (`src/frontend.py`)

**Purpose:** Interactive web interface for predictions.

### Health Check

```python
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    api_status = health.json().get("status") == "ok"
except:
    api_status = False

if not api_status:
    st.error("Backend API is not reachable")
```

Shows red/green indicator based on backend status.

### Single Prediction

```python
# Create UI inputs
attendance = st.slider("Attendance %", 0, 100, 85)
assignment = st.number_input("Assignment Avg", 0, 100, 70)
# ...

if st.button("Predict"):
    payload = {
        "attendance_percentage": attendance,
        "assignment_average": assignment,
        "internal_marks": internal,
        "previous_sem_gpa": gpa
    }
    
    # Call API
    resp = requests.post(f"{API_URL}/predict", json=payload)
    result = resp.json()
    
    # Display result
    if result['at_risk'] == 1:
        st.error(f"AT RISK ({result['risk_probability']:.1%})")
    else:
        st.success(f"SAFE ({result['risk_probability']:.1%})")
```

### Batch Upload

```python
uploaded_file = st.file_uploader("Choose CSV", type="csv")

if st.button("Analyze"):
    resp = requests.post(
        f"{API_URL}/predict_batch",
        files={"file": uploaded_file.getvalue()}
    )
    
    results_df = pd.DataFrame(resp.json())
    
    # Show summary
    st.metric("At Risk", results_df['Predicted_Risk'].sum())
    
    # Show table with highlighting
    st.dataframe(results_df.style.apply(
        lambda x: ['background-color: #ffcccb' if v == 1 else '' for v in x],
        subset=['Predicted_Risk']
    ))
    
    # Download button
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download", csv, "report.csv")
```

---

## How It All Works Together

### Training Flow
```
1. User runs: python main.py
2. main.py calls data_loader.load_data()
3. main.py creates Preprocessor and calls fit_transform()
4. main.py creates ModelTrainer and trains 4 models
5. main.py uses Evaluator to compare models
6. main.py saves best model + preprocessor to models/
7. Done! Artifacts ready for deployment
```

### Inference Flow (API)
```
1. User starts: uvicorn src.app:app --reload
2. app.py loads model + preprocessor from models/
3. User sends POST to /predict with student data
4. app.py uses preprocessor.transform() on new data
5. app.py uses model.predict() to get risk
6. app.py returns JSON with prediction
```

### Inference Flow (UI)
```
1. User starts Streamlit: streamlit run src/frontend.py
2. User enters data in web form
3. frontend.py sends HTTP POST to API
4. API returns prediction
5. frontend.py displays result with styling
```

---

## Key Takeaways

1. **Separation of Concerns**: Each module has single responsibility
2. **State Management**: Preprocessor saves fitted parameters
3. **Factory Pattern**: Easy to add new models
4. **Type Safety**: Pydantic validates API inputs
5. **Error Handling**: Try/except at all I/O boundaries
6. **Reproducibility**: random_state=42 everywhere
