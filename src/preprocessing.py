
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df: pd.DataFrame, target_column: str = 'at_risk'):
        """
        Preprocesses the data: drops ID, handles missing values (simple drop for now),
        scales features.
        
        Args:
            df (pd.DataFrame): Raw dataframe.
            target_column (str): Name of the target variable.
            
        Returns:
            X_train, X_test, y_train, y_test: Split data.
            full_processed_df: Dataframe with processed features + original ID (for reporting).
        """
        # Keep ID for later but drop for training
        if 'student_id' in df.columns:
            ids = df['student_id']
            X = df.drop(columns=['student_id', target_column])
        else:
            ids = pd.Series(range(len(df)), name='student_id')
            X = df.drop(columns=[target_column])
            
        y = df[target_column]

        # Handle missing values - for this dataset, we'll drop rows with NaNs if any, 
        # but the view showed no obvious gaps. We'll fill with median for robustness.
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Combine for full processed view
        full_processed_df = pd.concat([ids.reset_index(drop=True), X_scaled_df, y.reset_index(drop=True)], axis=1)

        return X_scaled, y, full_processed_df, X.columns

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
