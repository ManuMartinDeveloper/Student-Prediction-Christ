
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred)
        }

    @staticmethod
    def get_feature_importance(model, feature_names):
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        
        if importances is not None:
             feature_imp = pd.DataFrame({
                 'Feature': feature_names,
                 'Importance': importances
             }).sort_values(by='Importance', ascending=False)
             return feature_imp
        return None
