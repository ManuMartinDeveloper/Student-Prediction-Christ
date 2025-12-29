# Model Comparison Report

| Model | Accuracy | Precision | Recall | Test F1 | CV F1 (Mean) |
|-------|----------|-----------|--------|---------|--------------|
| LogisticRegression | 0.8750 | 0.9000 | 0.9474 | 0.9231 | 0.8957 |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9603 |
| SVM | 0.9167 | 0.9474 | 0.9474 | 0.9474 | 0.9328 |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9822 |

## Best Model Selection
The best performing model is **RandomForest**.
The model has been saved to `D:\Projects\Student Prediction\models\RandomForest.pkl`.

## Top Contributing Factors
| Feature               |   Importance |
|:----------------------|-------------:|
| attendance_percentage |     0.356758 |
| previous_sem_gpa      |     0.316245 |
| assignment_average    |     0.166485 |
| internal_marks        |     0.160512 |