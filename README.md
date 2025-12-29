# 🎓 Student Performance Risk Prediction System

An end-to-end Machine Learning solution that predicts student academic risk using attendance, assignment scores, internal marks, and previous semester GPA. This system includes a modular ML pipeline, REST API backend, and interactive web interface.

## 🌟 Features

- **Modular ML Pipeline**: Clean, maintainable code with separate modules for data loading, preprocessing, model training, and evaluation
- **Multiple Model Comparison**: Evaluates Logistic Regression, Random Forest, SVM, and Gradient Boosting with cross-validation
- **REST API**: FastAPI backend with endpoints for single and batch predictions
- **Interactive Web UI**: Streamlit frontend for easy CSV upload and prediction visualization
- **Comprehensive Reporting**: Auto-generated model comparison reports with feature importance analysis
- **Robust Validation**: 5-Fold Cross-Validation to ensure model generalization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory**
   ```powershell
   cd "d:\Projects\Student Prediction"
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

### Training the Model

Run the main pipeline to train models and generate artifacts:

```powershell
python main.py
```

This will:
- Load and preprocess the dataset
- Train 4 different models with cross-validation
- Save the best model (`RandomForest.pkl`) and preprocessor (`preprocessor.pkl`)
- Generate a comparison report in `reports/model_comparison.md`
- Create predictions CSV with all students' risk assessments

### Running the Application

#### 1. Start the Backend (Terminal 1)

```powershell
.\venv\Scripts\activate
uvicorn src.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`
- Interactive API docs: http://127.0.0.1:8000/docs

#### 2. Start the Frontend (Terminal 2)

```powershell
.\venv\Scripts\activate
streamlit run src/frontend.py
```

The web interface will open automatically in your browser (usually http://localhost:8501)

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | Test F1 | CV F1 (Mean) |
|-------|----------|-----------|--------|---------|--------------|
| **RandomForest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9603** |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9822 |
| SVM | 0.9167 | 0.9474 | 0.9474 | 0.9474 | 0.9328 |
| LogisticRegression | 0.8750 | 0.9000 | 0.9474 | 0.9231 | 0.8957 |

**Best Model**: RandomForest (selected based on test performance)

### Feature Importance

Top contributing factors to academic risk:
1. **Attendance Percentage** (35.7%)
2. **Previous Semester GPA** (31.6%)
3. **Assignment Average** (16.6%)
4. **Internal Marks** (16.1%)

## 🏗️ Project Structure

```
d:/Projects/Student Prediction/
├── src/
│   ├── __init__.py
│   ├── app.py                 # FastAPI backend
│   ├── frontend.py            # Streamlit UI
│   ├── data_loader.py         # Data loading module
│   ├── preprocessing.py       # Feature preprocessing
│   ├── models.py              # Model definitions
│   └── evaluation.py          # Metrics and evaluation
├── models/
│   ├── RandomForest.pkl       # Trained model
│   └── preprocessor.pkl       # Fitted preprocessor
├── reports/
│   └── model_comparison.md    # Performance report
├── notebooks/
│   └── student_risk_analysis.ipynb  # Step-by-step analysis
├── main.py                    # Training pipeline
├── test_api.py                # API testing script
├── requirements.txt           # Python dependencies
├── student_performance_risk_dataset.csv  # Dataset
└── student_risk_predictions.csv  # Generated predictions
```

## 🔧 Tech Stack

**Core ML Stack:**
- pandas - Data manipulation
- scikit-learn - Machine learning models and preprocessing
- joblib - Model serialization

**Web Stack:**
- FastAPI - REST API framework
- Uvicorn - ASGI server
- Streamlit - Interactive web UI
- pydantic - Data validation

**Development:**
- Jupyter Notebook - Interactive analysis
- python-multipart - File upload handling

## 📖 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `POST /predict`
Predict risk for a single student

**Request Body:**
```json
{
  "attendance_percentage": 85,
  "assignment_average": 70,
  "internal_marks": 65,
  "previous_sem_gpa": 7.5
}
```

**Response:**
```json
{
  "at_risk": 0,
  "risk_probability": 0.12
}
```

#### `POST /predict_batch`
Batch prediction from CSV upload

**Request:** Multipart form data with CSV file

**Response:** JSON array with predictions for all students

## 🎯 Usage Examples

### Single Student Prediction (Web UI)

1. Open the Streamlit interface
2. Select "Single Student Prediction"
3. Enter student details using sliders/inputs
4. Click "Predict Risk Status"
5. View instant results with risk probability

### Batch Analysis (Web UI)

1. Select "Batch Analysis (CSV)"
2. Upload your student data CSV file
3. Click "Analyze Batch"
4. View summary statistics and detailed predictions
5. Download the full report with predictions

### API Testing (Command Line)

```powershell
python test_api.py
```

Or use curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"attendance_percentage\":85,\"assignment_average\":70,\"internal_marks\":65,\"previous_sem_gpa\":7.5}"
```

## 📝 Dataset Format

The CSV file should contain the following columns:

| Column | Description | Range |
|--------|-------------|-------|
| `student_id` | Unique student identifier | String |
| `attendance_percentage` | Attendance rate | 0-100 |
| `assignment_average` | Average assignment score | 0-100 |
| `internal_marks` | Internal assessment marks | 0-100 |
| `previous_sem_gpa` | Previous semester GPA | 0.0-10.0 |
| `at_risk` | Target variable (optional for prediction) | 0 or 1 |

## 🔍 Future Enhancements

- [ ] Add authentication and user management
- [ ] Implement model retraining pipeline
- [ ] Add email notifications for at-risk students
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Add more sophisticated feature engineering
- [ ] Implement A/B testing for model versions
- [ ] Create mobile application

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for student success**
