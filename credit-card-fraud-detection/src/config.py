"""
Configuration file for Credit Card Fraud Detection project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
RAW_DATA_FILE = DATA_DIR / "creditcard.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.pkl"

# Model files
MODEL_FILE = MODELS_DIR / "fraud_detection_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering
FEATURES_TO_SCALE = ['Time', 'Amount']
PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
ALL_FEATURES = FEATURES_TO_SCALE + PCA_FEATURES
TARGET = 'Class'

# Model hyperparameters (default)
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'liblinear',
    'random_state': RANDOM_STATE,
    'max_iter': 1000
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# SMOTE parameters
SMOTE_PARAMS = {
    'sampling_strategy': 0.5,  # Minority class will be 50% of majority
    'random_state': RANDOM_STATE,
    'k_neighbors': 5
}

# Threshold for fraud detection
FRAUD_THRESHOLD = 0.5

# Streamlit app configuration
APP_TITLE = "💳 Credit Card Fraud Detection"
APP_DESCRIPTION = """
Bu uygulama, kredi kartı işlemlerinde dolandırıcılık tespiti yapmak için 
makine öğrenmesi modelini kullanır.
"""

# API configuration
API_TITLE = "Credit Card Fraud Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "REST API for credit card fraud detection"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
