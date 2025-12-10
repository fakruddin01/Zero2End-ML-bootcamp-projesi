"""
Inference module for fraud detection model
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import logging

from config import (
    MODEL_FILE, 
    SCALER_FILE, 
    FRAUD_THRESHOLD,
    ALL_FEATURES,
    FEATURES_TO_SCALE
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Fraud detection model wrapper for inference
    """
    
    def __init__(self, model_path: Path = MODEL_FILE, scaler_path: Path = SCALER_FILE):
        """
        Initialize the fraud detector
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.warning(f"Scaler file not found: {self.scaler_path}")
                
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def preprocess(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """
        Preprocess input data for prediction
        
        Args:
            data: Input data (DataFrame, dict, or list)
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Ensure all required features are present
        missing_features = set(ALL_FEATURES) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
            
        # Select only required features in correct order
        df = df[ALL_FEATURES]
        
        # Scale features if scaler is available
        if self.scaler is not None:
            df_scaled = df.copy()
            df_scaled[FEATURES_TO_SCALE] = self.scaler.transform(df[FEATURES_TO_SCALE])
            return df_scaled
        
        return df
    
    def predict(self, data: Union[pd.DataFrame, Dict, List], threshold: float = FRAUD_THRESHOLD) -> np.ndarray:
        """
        Predict fraud (binary classification)
        
        Args:
            data: Input data
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0: Normal, 1: Fraud)
        """
        if not self.is_loaded:
            self.load_model()
            
        # Preprocess data
        X = self.preprocess(data)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict, List]) -> np.ndarray:
        """
        Predict fraud probability
        
        Args:
            data: Input data
            
        Returns:
            Probability predictions for fraud class
        """
        if not self.is_loaded:
            self.load_model()
            
        # Preprocess data
        X = self.preprocess(data)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def predict_with_details(self, data: Union[pd.DataFrame, Dict, List], 
                            threshold: float = FRAUD_THRESHOLD) -> List[Dict]:
        """
        Predict with detailed results
        
        Args:
            data: Input data
            threshold: Classification threshold
            
        Returns:
            List of dictionaries with prediction details
        """
        if not self.is_loaded:
            self.load_model()
            
        # Get predictions and probabilities
        predictions = self.predict(data, threshold)
        probabilities = self.predict_proba(data)
        
        # Create detailed results
        results = []
        for pred, prob in zip(predictions, probabilities):
            result = {
                'prediction': int(pred),
                'prediction_label': 'Fraud' if pred == 1 else 'Normal',
                'fraud_probability': float(prob),
                'confidence': float(prob) if pred == 1 else float(1 - prob),
                'risk_level': self._get_risk_level(prob)
            }
            results.append(result)
            
        return results
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Determine risk level based on fraud probability
        
        Args:
            probability: Fraud probability
            
        Returns:
            Risk level string
        """
        if probability < 0.3:
            return 'Low'
        elif probability < 0.6:
            return 'Medium'
        elif probability < 0.8:
            return 'High'
        else:
            return 'Critical'


# Convenience function for quick predictions
def predict_fraud(data: Union[pd.DataFrame, Dict, List], 
                 model_path: Path = MODEL_FILE,
                 scaler_path: Path = SCALER_FILE,
                 threshold: float = FRAUD_THRESHOLD) -> List[Dict]:
    """
    Quick fraud prediction function
    
    Args:
        data: Input data
        model_path: Path to model file
        scaler_path: Path to scaler file
        threshold: Classification threshold
        
    Returns:
        List of prediction dictionaries
    """
    detector = FraudDetector(model_path, scaler_path)
    return detector.predict_with_details(data, threshold)


if __name__ == "__main__":
    # Example usage
    sample_transaction = {
        'Time': 0,
        'Amount': 149.62,
        'V1': -1.359807,
        'V2': -0.072781,
        'V3': 2.536347,
        'V4': 1.378155,
        'V5': -0.338321,
        'V6': 0.462388,
        'V7': 0.239599,
        'V8': 0.098698,
        'V9': 0.363787,
        'V10': 0.090794,
        'V11': -0.551600,
        'V12': -0.617801,
        'V13': -0.991390,
        'V14': -0.311169,
        'V15': 1.468177,
        'V16': -0.470401,
        'V17': 0.207971,
        'V18': 0.025791,
        'V19': 0.403993,
        'V20': 0.251412,
        'V21': -0.018307,
        'V22': 0.277838,
        'V23': -0.110474,
        'V24': 0.066928,
        'V25': 0.128539,
        'V26': -0.189115,
        'V27': 0.133558,
        'V28': -0.021053
    }
    
    print("Sample transaction prediction:")
    print(sample_transaction)
    
    # Note: This will fail until model is trained
    # results = predict_fraud(sample_transaction)
    # print("\nPrediction results:")
    # print(results)
