
import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import Preprocessor
from src.models import ModelTrainer
from src.evaluation import Evaluator

def main():
    # Paths
    DATA_PATH = os.path.join(os.getcwd(), 'student_performance_risk_dataset.csv')
    REPORT_PATH = os.path.join(os.getcwd(), 'reports', 'model_comparison.md')
    PREDICTIONS_PATH = os.path.join(os.getcwd(), 'student_risk_predictions.csv')
    
    # 1. Load Data
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(e)
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    preprocessor = Preprocessor()
    X_scaled, y, full_processed_df, feature_names = preprocessor.preprocess(df)
    
    # Split
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
    
    # 3. Model Training & Evaluation
    trainer = ModelTrainer()
    evaluator = Evaluator()
    
    models_to_train = ['LogisticRegression', 'RandomForest', 'SVM', 'GradientBoosting']
    results = {}
    best_f1 = -1
    best_model_name = ""
    
    print("\nTraining models...")
    for model_name in models_to_train:
        trainer.train_model(model_name, X_train, y_train)
        y_pred = trainer.predict(model_name, X_test)
        metrics = evaluator.evaluate(y_test, y_pred)
        results[model_name] = metrics
        
        print(f"  {model_name}: {metrics}")
        
        if metrics['F1-Score'] > best_f1:
            best_f1 = metrics['F1-Score']
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")

    # 4. Generate Report
    print("Generating report...")
    with open(REPORT_PATH, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        for model_name, metrics in results.items():
            f.write(f"| {model_name} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1-Score']:.4f} |\n")
        
        f.write(f"\n## Best Model Selection\n")
        f.write(f"The best performing model is **{best_model_name}** with an F1-Score of **{best_f1:.4f}**.\n")
        
        # Feature Importance for Best Model
        best_model = trainer.models[best_model_name]
        feature_imp = evaluator.get_feature_importance(best_model, feature_names)
        if feature_imp is not None:
            f.write("\n## Top Contributing Factors\n")
            f.write(feature_imp.to_markdown(index=False))
    
    print(f"Report saved to {REPORT_PATH}")
    
    # 5. Save Full Predictions for Best Model
    # We want to predict on the FULL dataset to give a complete list as requested
    # Note: In a real scenario, predicting on training data is 'cheating' for evaluation, 
    # but for a 'risk list' output it's common to want scores for everyone.
    # We already have scaled full data 'X_scaled'
    
    print("Generating full prediction list...")
    full_preds = trainer.predict(best_model_name, X_scaled)
    full_probs = trainer.predict_proba(best_model_name, X_scaled)
    
    # Add predictions to original dataframe (or a copy) to preserve IDs
    output_df = df.copy()
    output_df['Predicted_Risk'] = full_preds
    if full_probs is not None:
        output_df['Risk_Probability'] = full_probs
        
    output_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")
    print(output_df.head())

if __name__ == "__main__":
    main()
