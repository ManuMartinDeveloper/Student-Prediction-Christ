
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.target_stats = None # Optional: derived stats

    def _get_X_y(self, df: pd.DataFrame, target_column: str = 'at_risk', training: bool = True):
        # Drop ID if present
        if 'student_id' in df.columns:
            X = df.drop(columns=['student_id'])
        else:
            X = df.copy()
            
        if training and target_column in X.columns:
            y = X[target_column]
            X = X.drop(columns=[target_column])
        else:
            y = None
            if target_column in X.columns:
                 X = X.drop(columns=[target_column])
        
        return X, y

    def fit_transform(self, df: pd.DataFrame, target_column: str = 'at_risk'):
        """
        Fits the imputer and scaler on the data and transforms it.
        """
        X, y = self._get_X_y(df, target_column, training=True)
        
        # Fit Imputer
        X_imputed = self.imputer.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        # Fit Scaler
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Return full view for reporting if needed, but mainly X and y
        full_processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True) if y is not None else pd.DataFrame()], axis=1)
        
        return X_scaled, y, full_processed_df, self.feature_names

    def transform(self, df: pd.DataFrame):
        """
        Transforms new data using fitted imputer and scaler.
        """
        X, _ = self._get_X_y(df, training=False)
        
        # Ensure columns match
        # (In a real scenario, we'd handle missing/extra columns more robustly)
        
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
