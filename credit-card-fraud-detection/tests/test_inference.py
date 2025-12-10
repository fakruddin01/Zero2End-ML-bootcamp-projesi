"""
Unit tests for inference module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import FraudDetector


class TestFraudDetector:
    """Test cases for FraudDetector class"""
    
    @pytest.fixture
    def sample_transaction(self):
        """Create a sample transaction for testing"""
        return {
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
    
    def test_detector_initialization(self):
        """Test FraudDetector initialization"""
        detector = FraudDetector()
        assert detector.model is None
        assert detector.scaler is None
        assert not detector.is_loaded
    
    def test_preprocess_dict(self, sample_transaction):
        """Test preprocessing with dictionary input"""
        detector = FraudDetector()
        df = detector.preprocess(sample_transaction)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in ['Time', 'Amount'])
    
    def test_preprocess_dataframe(self, sample_transaction):
        """Test preprocessing with DataFrame input"""
        detector = FraudDetector()
        df_input = pd.DataFrame([sample_transaction])
        df = detector.preprocess(df_input)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    def test_risk_level_calculation(self):
        """Test risk level calculation"""
        assert FraudDetector._get_risk_level(0.2) == 'Low'
        assert FraudDetector._get_risk_level(0.5) == 'Medium'
        assert FraudDetector._get_risk_level(0.7) == 'High'
        assert FraudDetector._get_risk_level(0.9) == 'Critical'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
