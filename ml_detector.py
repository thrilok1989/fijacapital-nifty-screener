import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, accuracy_score, 
                           precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix)
import joblib
from datetime import datetime, timedelta
import talib
from typing import Tuple, Dict, List, Optional, Any
import asyncio
import logging
from config import Config

logger = logging.getLogger(__name__)

class ExpirySpikeMLDetector:
    """Machine Learning model for expiry spike detection"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_columns = []
        self.model_path = model_path or Config.ML_MODEL_PATH
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize XGBoost model with optimized parameters"""
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
            eval_metric='auc',
            early_stopping_rounds=20,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        logger.info("Initialized new XGBoost model")
    
    def extract_features(self, historical_data: pd.DataFrame, 
                        option_data: Dict, gamma_data: Dict) -> np.ndarray:
        """Extract comprehensive features for ML model"""
        features = []
        
        # 1. Price Action Features
        if len(historical_data) >= 50:
            close_prices = historical_data['close'].values
            high_prices = historical_data['high'].values
            low_prices = historical_data['low'].values
            volume = historical_data['volume'].values
            
            # Technical Indicators
            rsi = talib.RSI(close_prices, timeperiod=14)[-1]
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            macd_value = macd[-1] - macd_signal[-1]
            
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            atr_percent = atr / close_prices[-1]
            
            # Volume indicators
            volume_sma = talib.SMA(volume, timeperiod=20)[-1]
            volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 0
            
            obv = talib.OBV(close_prices, volume)[-1]
            
            features.extend([
                rsi / 100,  # Normalize to 0-1
                macd_value / close_prices[-1],  # Normalize
                bb_position,
                atr_percent,
                volume_ratio,
                obv / 1e6  # Scale down
            ])
        
        # 2. Option Chain Features
        underlying_price = option_data.get('underlying_price', 0)
        
        # Volume metrics
        total_volume = option_data.get('total_volume', 0)
        if len(historical_data) > 0:
            avg_underlying_volume = historical_data['volume'].mean()
            volume_ratio_opt = total_volume / avg_underlying_volume if avg_underlying_volume > 0 else 0
            features.append(volume_ratio_opt)
        
        # OI metrics
        total_oi = option_data.get('total_oi', 0)
        features.append(np.log1p(total_oi) / 20)  # Normalized log scale
        
        # Put-Call Ratios
        total_call_volume = sum([c['volume'] for c in option_data.get('calls', [])])
        total_put_volume = sum([p['volume'] for p in option_data.get('puts', [])])
        total_call_oi = sum([c['oi'] for c in option_data.get('calls', [])])
        total_put_oi = sum([p['oi'] for p in option_data.get('puts', [])])
        
        pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 1
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1
        
        features.extend([pcr_volume, pcr_oi])
        
        # IV Features
        calls_iv = [c['iv'] for c in option_data.get('calls', []) if c['iv'] > 0]
        puts_iv = [p['iv'] for p in option_data.get('puts', []) if p['iv'] > 0]
        
        if calls_iv and puts_iv:
            avg_iv = np.mean(calls_iv + puts_iv)
            iv_skew = (np.mean(calls_iv) - np.mean(puts_iv)) / avg_iv if avg_iv > 0 else 0
            iv_std = np.std(calls_iv + puts_iv)
            
            features.extend([avg_iv / 100, iv_skew, iv_std / 100])
        else:
            features.extend([0, 0, 0])
        
        # 3. Gamma Features
        if gamma_data:
            gamma_exposure = gamma_data.get('gamma_exposure', 0)
            total_gamma = gamma_data.get('total_gamma', 0)
            
            # Normalize gamma exposure by underlying price and OI
            gamma_exposure_norm = gamma_exposure / (underlying_price * total_oi) if total_oi > 0 else 0
            
            call_gamma = gamma_data.get('call_gamma', 0)
            put_gamma = gamma_data.get('put_gamma', 0)
            gamma_ratio = call_gamma / put_gamma if put_gamma > 0 else 1
            
            features.extend([
                gamma_exposure_norm,
                np.log1p(abs(gamma_exposure)) / 20,
                gamma_ratio
            ])
        else:
            features.extend([0, 0, 1])
        
        # 4. Time Features
        expiry_date = datetime.strptime(option_data.get('expiry_date', ''), '%Y-%m-%d')
        days_to_expiry = (expiry_date - datetime.now()).days
        
        # Days to expiry features (non-linear transformation)
        features.extend([
            days_to_expiry / 30,  # Normalized to months
            1 / (days_to_expiry + 1),  # Inverse for near expiry
            np.log1p(days_to_expiry) / 5  # Log scale
        ])
        
        # 5. Market Microstructure Features
        # Bid-ask spread (average)
        call_spreads = []
        put_spreads = []
        
        for call in option_data.get('calls', []):
            if call.get('ask', 0) > 0 and call.get('bid', 0) > 0:
                spread = (call['ask'] - call['bid']) / call['ask']
                call_spreads.append(spread)
        
        for put in option_data.get('puts', []):
            if put.get('ask', 0) > 0 and put.get('bid', 0) > 0:
                spread = (put['ask'] - put['bid']) / put['ask']
                put_spreads.append(spread)
        
        avg_call_spread = np.mean(call_spreads) if call_spreads else 0
        avg_put_spread = np.mean(put_spreads) if put_spreads else 0
        
        features.extend([avg_call_spread, avg_put_spread])
        
        # 6. Statistical Features
        # Option chain concentration
        strike_prices = [c['strike'] for c in option_data.get('calls', [])]
        if strike_prices:
            atm_strike = min(strike_prices, key=lambda x: abs(x - underlying_price))
            features.append(abs(underlying_price - atm_strike) / underlying_price)
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Handle NaN values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features_array
    
    async def train_model(self, training_data: pd.DataFrame, 
                         validation_split: float = 0.2):
        """Train the ML model with time series cross-validation"""
        
        if 'is_spike' not in training_data.columns:
            raise ValueError("Training data must contain 'is_spike' column")
        
        # Separate features and target
        X = training_data.drop(['is_spike', 'timestamp'], axis=1, errors='ignore')
        y = training_data['is_spike']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        scores = []
        feature_importances = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val_scaled)
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            
            scores.append({
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall
            })
            
            # Collect feature importances
            importances = self.model.feature_importances_
            feature_importances.append(importances)
        
        # Average scores
        avg_scores = {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}
        
        # Set feature columns
        self.feature_columns = X.columns.tolist()
        
        # Calculate average feature importance
        avg_importances = np.mean(feature_importances, axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': avg_importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Model trained. Average AUC: {avg_scores['auc']:.4f}")
        logger.info(f"Feature importances:\n{feature_importance_df.head(10)}")
        
        return avg_scores, feature_importance_df
    
    def predict_spike(self, features: np.ndarray) -> Tuple[bool, float]:
        """Predict if there's a spike"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Use adaptive threshold based on model confidence
        threshold = Config.SPIKE_THRESHOLD
        prediction = probability > threshold
        
        return bool(prediction), float(probability)
    
    def save_model(self, path: str = None):
        """Save model and scaler"""
        save_path = path or self.model_path
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': {
                'saved_at': datetime.now().isoformat(),
                'model_type': 'XGBoost',
                'version': '1.0'
            }
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load model and scaler"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.initialize_model()
    
    async def create_training_data(self, symbol: str, days_back: int = 90) -> pd.DataFrame:
        """Create training data from historical records"""
        try:
            # This would fetch historical data and create labeled examples
            # For now, return empty DataFrame
            logger.warning("Training data creation not fully implemented")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return pd.DataFrame()
    
    def explain_prediction(self, features: np.ndarray, 
                          feature_names: List[str] = None) -> Dict:
        """Explain model prediction with feature contributions"""
        if feature_names is None:
            feature_names = self.feature_columns
        
        if self.model is None or len(feature_names) != features.shape[1]:
            return {}
        
        # Get feature contributions (simplified SHAP-like)
        feature_contributions = {}
        
        # For tree-based models, we can get feature importance per prediction
        try:
            # Get feature importance scores
            importances = self.model.feature_importances_
            
            # Normalize features
            features_scaled = self.scaler.transform(features)
            
            # Simple heuristic: contribution = importance * feature value
            for i, (name, imp, val) in enumerate(zip(feature_names, importances, features_scaled[0])):
                contribution = imp * abs(val)
                feature_contributions[name] = {
                    'importance': float(imp),
                    'value': float(val),
                    'contribution': float(contribution),
                    'scaled_value': float(val)
                }
            
            # Sort by contribution
            sorted_contributions = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]['contribution']),
                reverse=True
            )
            
            return {
                'top_contributors': dict(sorted_contributions[:10]),
                'all_features': feature_contributions
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {}
