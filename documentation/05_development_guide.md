# Development Guide

## Quick Setup
```powershell
cd "d:\Projects\Student Prediction"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Adding a New Model
1. Edit `src/models.py` - add to `ModelFactory.get_model()`
2. Edit `main.py` - add model name to `models_to_train` list
3. Run: `python main.py`

## Testing
```powershell
python test_api.py  # Test API endpoints
```

## Code Style
- Follow PEP 8
- Add docstrings
- Use type hints

## Common Tasks

### Retrain Model
```powershell
python main.py
```

### Start Development Server
```powershell
uvicorn src.app:app --reload
```

### Run Frontend
```powershell
streamlit run src/frontend.py
```
