
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import joblib

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
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

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = ""

    def train_model(self, model_name: str, X_train, y_train, **kwargs):
        model = ModelFactory.get_model(model_name, **kwargs)
        model.fit(X_train, y_train)
        self.models[model_name] = model
        print(f"Trained {model_name}")
        return model

    def predict(self, model_name: str, X_test):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained.")
        return self.models[model_name].predict(X_test)
    
    def predict_proba(self, model_name: str, X_test):
        if model_name not in self.models:
             raise ValueError(f"Model {model_name} has not been trained.")
        if hasattr(self.models[model_name], "predict_proba"):
            return self.models[model_name].predict_proba(X_test)[:, 1]
        else:
            return None

    def save_model(self, model_name: str, filepath: str):
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Saved {model_name} to {filepath}")
        else:
             print(f"Model {model_name} not found to save.")
