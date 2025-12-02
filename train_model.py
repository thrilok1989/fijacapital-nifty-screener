# train_model.py
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from ml_detector import ExpirySpikeMLDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_training_data():
    """
    Create synthetic training data for initial setup.
    In production, replace with real historical data.
    """
    logger.info("Creating training data...")
    
    n_samples = 1000
    n_features = 20  # Match the number of features in ml_detector.py
    
    # Create synthetic features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic targets with some patterns
    # Spike when features 0, 5, 10 are high and feature 15 is low
    spike_condition = (
        (X[:, 0] > 1.0) &  # High RSI
        (X[:, 5] > 0.5) &  # High volume ratio
        (X[:, 10] > 0.3) & # High gamma exposure
        (X[:, 15] < -0.2)  # Negative IV skew
    )
    
    # Add some noise
    y = spike_condition.astype(int)
    noise = np.random.rand(n_samples) < 0.1  # 10% noise
    y[noise] = 1 - y[noise]
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['is_spike'] = y
    df['timestamp'] = pd.date_range(
        end=pd.Timestamp.now(), 
        periods=n_samples, 
        freq='5min'
    )
    
    # Add some realistic column names matching your features
    realistic_features = [
        'rsi_normalized', 'macd_normalized', 'bb_position', 'atr_percent',
        'volume_ratio_opt', 'log_oi_normalized', 'pcr_volume', 'pcr_oi',
        'avg_iv', 'iv_skew', 'iv_std', 'gamma_exposure_norm',
        'log_gamma_exposure', 'gamma_ratio', 'days_to_expiry_normalized',
        'inverse_days_to_expiry', 'log_days_to_expiry', 'avg_call_spread',
        'avg_put_spread', 'atm_distance_percent'
    ]
    
    if len(df.columns) - 2 == len(realistic_features):  # minus is_spike and timestamp
        df = df.rename(columns=dict(zip(feature_names, realistic_features)))
    
    logger.info(f"Created training data with {n_samples} samples")
    logger.info(f"Spike ratio: {y.mean():.2%}")
    
    return df

def train_initial_model():
    """Train initial ML model"""
    logger.info("Training initial model...")
    
    # Create training data
    training_data = create_training_data()
    
    # Initialize and train model
    detector = ExpirySpikeMLDetector()
    
    # Train model
    scores, feature_importance = await detector.train_model(training_data)
    
    logger.info(f"Model trained with accuracy: {scores.get('accuracy', 0):.2%}")
    logger.info(f"AUC: {scores.get('auc', 0):.2%}")
    
    # Save model
    detector.save_model('models/spike_detector.pkl')
    
    # Save feature importance
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    logger.info("✅ Model saved to models/spike_detector.pkl")
    
    return detector

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_initial_model())
