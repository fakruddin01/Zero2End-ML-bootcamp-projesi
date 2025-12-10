"""
ML Pipeline for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from config import (
    RAW_DATA_FILE,
    MODEL_FILE,
    SCALER_FILE,
    ALL_FEATURES,
    TARGET,
    FEATURES_TO_SCALE,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    SMOTE_PARAMS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Complete ML pipeline for fraud detection
    """
    
    def __init__(self, data_path: Path = RAW_DATA_FILE):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the raw data file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset"""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded: {self.df.shape}")
        
        return self.df
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Basic data exploration
        
        Returns:
            Dictionary with exploration results
        """
        if self.df is None:
            self.load_data()
        
        exploration = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'class_distribution': self.df[TARGET].value_counts().to_dict(),
            'fraud_percentage': (self.df[TARGET].sum() / len(self.df)) * 100,
            'basic_stats': self.df.describe().to_dict()
        }
        
        logger.info(f"Class distribution: {exploration['class_distribution']}")
        logger.info(f"Fraud percentage: {exploration['fraud_percentage']:.4f}%")
        
        return exploration
    
    def preprocess_data(self, apply_smote: bool = True) -> Tuple:
        """
        Preprocess the data
        
        Args:
            apply_smote: Whether to apply SMOTE for balancing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = self.df[ALL_FEATURES].copy()
        y = self.df[TARGET].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[FEATURES_TO_SCALE] = self.scaler.fit_transform(X_train[FEATURES_TO_SCALE])
        X_test_scaled[FEATURES_TO_SCALE] = self.scaler.transform(X_test[FEATURES_TO_SCALE])
        
        logger.info("Features scaled")
        
        # Apply SMOTE if requested
        if apply_smote:
            logger.info("Applying SMOTE...")
            smote = SMOTE(**SMOTE_PARAMS)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"After SMOTE: {X_train_scaled.shape}")
            logger.info(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model=None, **model_params) -> Any:
        """
        Train the model
        
        Args:
            model: Model instance (if None, uses LogisticRegression)
            **model_params: Model parameters
            
        Returns:
            Trained model
        """
        if self.X_train is None:
            self.preprocess_data()
        
        if model is None:
            logger.info("Training Logistic Regression model...")
            self.model = LogisticRegression(**model_params) if model_params else LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            )
        else:
            logger.info(f"Training {type(model).__name__} model...")
            self.model = model
        
        # Train
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model training completed")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, 
            self.X_train, 
            self.y_train, 
            cv=CV_FOLDS,
            scoring='roc_auc'
        )
        logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.model
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        evaluation = {
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'roc_auc_score': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        logger.info(f"ROC-AUC Score: {evaluation['roc_auc_score']:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(self.y_test, y_pred))
        
        return evaluation
    
    def save_model(self, model_path: Path = MODEL_FILE, scaler_path: Path = SCALER_FILE):
        """
        Save the trained model and scaler
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create directories if they don't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    def run_full_pipeline(self, model=None, apply_smote: bool = True, **model_params) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            model: Model to train (if None, uses LogisticRegression)
            apply_smote: Whether to apply SMOTE
            **model_params: Model parameters
            
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 50)
        logger.info("Starting Full ML Pipeline")
        logger.info("=" * 50)
        
        # Load and explore
        self.load_data()
        exploration = self.explore_data()
        
        # Preprocess
        self.preprocess_data(apply_smote=apply_smote)
        
        # Train
        self.train_model(model=model, **model_params)
        
        # Evaluate
        evaluation = self.evaluate_model()
        
        # Save
        self.save_model()
        
        logger.info("=" * 50)
        logger.info("Pipeline Completed Successfully")
        logger.info("=" * 50)
        
        return {
            'exploration': exploration,
            'evaluation': evaluation
        }


if __name__ == "__main__":
    # Example usage
    pipeline = FraudDetectionPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(apply_smote=True)
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"\nROC-AUC Score: {results['evaluation']['roc_auc_score']:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(results['evaluation']['classification_report']).transpose())
