"""
Synthetic Data Generator for Student Risk Prediction System

This script generates realistic synthetic student performance data based on 
statistical patterns observed in the existing dataset. It can be used for 
testing, model validation, and data augmentation.

Usage:
    python generate_synthetic_data.py --num_samples 500 --output Dataset/synthetic_data.csv
    python generate_synthetic_data.py --num_samples 1000 --balance 0.5 --seed 42
"""

import pandas as pd
import numpy as np
import argparse
import os
from typing import Tuple


class SyntheticDataGenerator:
    """
    Generates synthetic student performance data with realistic distributions
    and correlations based on the original dataset patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the generator with a random seed for reproducibility.
        
        Args:
            seed: Random seed for numpy random number generator
        """
        self.seed = seed
        np.random.seed(seed)
        
    def analyze_original_data(self, filepath: str) -> dict:
        """
        Analyze the original dataset to extract statistical patterns.
        
        Args:
            filepath: Path to the original CSV file
            
        Returns:
            Dictionary containing statistical parameters
        """
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            stats = {
                'attendance_mean': df['attendance_percentage'].mean(),
                'attendance_std': df['attendance_percentage'].std(),
                'assignment_mean': df['assignment_average'].mean(),
                'assignment_std': df['assignment_average'].std(),
                'internal_mean': df['internal_marks'].mean(),
                'internal_std': df['internal_marks'].std(),
                'gpa_mean': df['previous_sem_gpa'].mean(),
                'gpa_std': df['previous_sem_gpa'].std(),
                'at_risk_ratio': df['at_risk'].mean()
            }
            print(f"Analyzed original dataset: {filepath}")
            print(f"  At-risk ratio: {stats['at_risk_ratio']:.2%}")
            return stats
        else:
            # Default parameters based on typical student performance distributions
            print("Original dataset not found. Using default parameters.")
            return {
                'attendance_mean': 75.0,
                'attendance_std': 12.0,
                'assignment_mean': 68.0,
                'assignment_std': 18.0,
                'internal_mean': 65.0,
                'internal_std': 16.0,
                'gpa_mean': 6.5,
                'gpa_std': 1.8,
                'at_risk_ratio': 0.7
            }
    
    def generate_correlated_features(self, 
                                     n_samples: int,
                                     stats: dict) -> pd.DataFrame:
        """
        Generate correlated features that mimic real student performance patterns.
        
        Args:
            n_samples: Number of samples to generate
            stats: Statistical parameters from original data
            
        Returns:
            DataFrame with generated features
        """
        # Generate attendance with realistic distribution
        attendance = np.random.normal(stats['attendance_mean'], 
                                     stats['attendance_std'], 
                                     n_samples)
        attendance = np.clip(attendance, 50, 100).round(0)
        
        # Generate assignment average with some correlation to attendance
        assignment_base = np.random.normal(stats['assignment_mean'], 
                                          stats['assignment_std'], 
                                          n_samples)
        # Positive correlation with attendance (better attendance -> better assignments)
        attendance_effect = (attendance - 75) * 0.3
        assignment = assignment_base + attendance_effect
        assignment = np.clip(assignment, 35, 100).round(0)
        
        # Generate internal marks with correlation to both attendance and assignments
        internal_base = np.random.normal(stats['internal_mean'], 
                                        stats['internal_std'], 
                                        n_samples)
        combined_effect = (attendance - 75) * 0.2 + (assignment - 68) * 0.15
        internal = internal_base + combined_effect
        internal = np.clip(internal, 35, 100).round(0)
        
        # Generate GPA with some independence but slight correlation to performance
        gpa_base = np.random.normal(stats['gpa_mean'], 
                                   stats['gpa_std'], 
                                   n_samples)
        gpa = np.clip(gpa_base, 4.0, 10.0).round(2)
        
        # Create DataFrame
        df = pd.DataFrame({
            'attendance_percentage': attendance.astype(int),
            'assignment_average': assignment.astype(int),
            'internal_marks': internal.astype(int),
            'previous_sem_gpa': gpa
        })
        
        return df
    
    def assign_risk_labels(self, df: pd.DataFrame, target_ratio: float = None) -> pd.Series:
        """
        Assign at_risk labels based on realistic rules and thresholds.
        
        Rules for being at risk:
        - Low attendance (< 70%) significantly increases risk
        - Low assignment average (< 55) increases risk
        - Low internal marks (< 60) increases risk
        - Very high GPA (> 8.5) can offset some risk factors
        
        Args:
            df: DataFrame with features
            target_ratio: Target ratio of at-risk students (None = use natural distribution)
            
        Returns:
            Series of binary labels (0 = not at risk, 1 = at risk)
        """
        # Calculate risk score (higher = more at risk)
        risk_score = 0.0
        
        # Attendance component (40% weight)
        risk_score += (100 - df['attendance_percentage']) * 0.4
        
        # Assignment component (30% weight)
        risk_score += (100 - df['assignment_average']) * 0.3
        
        # Internal marks component (20% weight)
        risk_score += (100 - df['internal_marks']) * 0.2
        
        # GPA component (10% weight, inverse - higher GPA reduces risk)
        risk_score += (10 - df['previous_sem_gpa']) * 1.0
        
        if target_ratio is not None:
            # Use threshold to achieve target ratio
            threshold = np.percentile(risk_score, (1 - target_ratio) * 100)
            at_risk = (risk_score >= threshold).astype(int)
        else:
            # Use a reasonable threshold (score > 50 indicates at risk)
            at_risk = (risk_score > 50).astype(int)
        
        return at_risk
    
    def generate_data(self, 
                     n_samples: int,
                     start_id: int = 1000,
                     balance_ratio: float = None,
                     original_data_path: str = None) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_samples: Number of samples to generate
            start_id: Starting student ID number
            balance_ratio: Ratio of at-risk students (None = natural distribution)
            original_data_path: Path to original data for statistical analysis
            
        Returns:
            Complete synthetic dataset
        """
        print(f"\nGenerating {n_samples} synthetic student records...")
        
        # Analyze original data if available
        if original_data_path and os.path.exists(original_data_path):
            stats = self.analyze_original_data(original_data_path)
            if balance_ratio is None:
                balance_ratio = stats['at_risk_ratio']
        else:
            stats = self.analyze_original_data(None)
            if balance_ratio is None:
                balance_ratio = 0.7  # Default 70% at-risk
        
        # Generate student IDs
        student_ids = [f'S{start_id + i}' for i in range(n_samples)]
        
        # Generate features
        df = self.generate_correlated_features(n_samples, stats)
        
        # Add student IDs
        df.insert(0, 'student_id', student_ids)
        
        # Assign risk labels
        df['at_risk'] = self.assign_risk_labels(df, balance_ratio)
        
        # Validation
        self._validate_data(df)
        
        print(f"[OK] Generated {n_samples} records")
        print(f"  At-risk students: {df['at_risk'].sum()} ({df['at_risk'].mean():.1%})")
        print(f"  Not at-risk students: {(1-df['at_risk']).sum()} ({(1-df['at_risk'].mean()):.1%})")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the generated data for quality and consistency.
        
        Args:
            df: Generated dataset
            
        Raises:
            ValueError: If data validation fails
        """
        # Check for required columns
        required_cols = ['student_id', 'attendance_percentage', 'assignment_average',
                        'internal_marks', 'previous_sem_gpa', 'at_risk']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check data ranges
        assert df['attendance_percentage'].between(50, 100).all(), "Invalid attendance values"
        assert df['assignment_average'].between(35, 100).all(), "Invalid assignment values"
        assert df['internal_marks'].between(35, 100).all(), "Invalid internal marks"
        assert df['previous_sem_gpa'].between(4.0, 10.0).all(), "Invalid GPA values"
        assert df['at_risk'].isin([0, 1]).all(), "Invalid at_risk labels"
        
        # Check for duplicates
        assert df['student_id'].is_unique, "Duplicate student IDs found"
        
        print("[OK] Data validation passed")
    
    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save the generated data to a CSV file.
        
        Args:
            df: Generated dataset
            output_path: Path to save the CSV file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved to {output_path}")


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic student performance data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 500 samples with default settings
  python generate_synthetic_data.py --num_samples 500
  
  # Generate 1000 samples with 50-50 class balance
  python generate_synthetic_data.py --num_samples 1000 --balance 0.5
  
  # Generate data with custom output path and seed
  python generate_synthetic_data.py --num_samples 2000 --output Dataset/test_data.csv --seed 123
        """
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=500,
        help='Number of synthetic samples to generate (default: 500)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='Dataset/synthetic_student_data.csv',
        help='Output CSV file path (default: Dataset/synthetic_student_data.csv)'
    )
    
    parser.add_argument(
        '--balance',
        type=float,
        default=None,
        help='Ratio of at-risk students (0.0-1.0). If not specified, uses natural distribution'
    )
    
    parser.add_argument(
        '--start_id',
        type=int,
        default=2000,
        help='Starting student ID number (default: 2000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--original_data',
        type=str,
        default='Dataset/student_performance_risk_dataset.csv',
        help='Path to original data for statistical analysis'
    )
    
    args = parser.parse_args()
    
    # Validate balance ratio
    if args.balance is not None and not (0.0 <= args.balance <= 1.0):
        parser.error("--balance must be between 0.0 and 1.0")
    
    # Create generator
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate data
    df = generator.generate_data(
        n_samples=args.num_samples,
        start_id=args.start_id,
        balance_ratio=args.balance,
        original_data_path=args.original_data
    )
    
    # Save data
    generator.save_data(df, args.output)
    
    print("\n" + "="*60)
    print("Synthetic data generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
