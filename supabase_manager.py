import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from supabase import create_client, Client
from config import Config

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages all database operations with Supabase"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.supabase_url = supabase_url or Config.SUPABASE_URL
        self.supabase_key = supabase_key or Config.SUPABASE_KEY
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key are required")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
    async def store_option_chain(self, chain_data: Dict) -> bool:
        """Store option chain data in database"""
        try:
            records = []
            timestamp = datetime.now().isoformat()
            
            # Prepare call records
            for call in chain_data.get('calls', []):
                record = {
                    'symbol': chain_data['symbol'],
                    'strike_price': float(call['strike']),
                    'expiry_date': chain_data['expiry_date'],
                    'option_type': 'CE',
                    'ltp': float(call.get('ltp', 0)),
                    'volume': int(call.get('volume', 0)),
                    'oi': int(call.get('oi', 0)),
                    'change_in_oi': int(call.get('change_in_oi', 0)),
                    'iv': float(call.get('iv', 0)),
                    'delta': float(call.get('delta', 0)),
                    'gamma': float(call.get('gamma', 0)),
                    'theta': float(call.get('theta', 0)),
                    'vega': float(call.get('vega', 0)),
                    'timestamp': timestamp
                }
                records.append(record)
            
            # Prepare put records
            for put in chain_data.get('puts', []):
                record = {
                    'symbol': chain_data['symbol'],
                    'strike_price': float(put['strike']),
                    'expiry_date': chain_data['expiry_date'],
                    'option_type': 'PE',
                    'ltp': float(put.get('ltp', 0)),
                    'volume': int(put.get('volume', 0)),
                    'oi': int(put.get('oi', 0)),
                    'change_in_oi': int(put.get('change_in_oi', 0)),
                    'iv': float(put.get('iv', 0)),
                    'delta': float(put.get('delta', 0)),
                    'gamma': float(put.get('gamma', 0)),
                    'theta': float(put.get('theta', 0)),
                    'vega': float(put.get('vega', 0)),
                    'timestamp': timestamp
                }
                records.append(record)
            
            # Batch insert
            if records:
                # Split into batches to avoid payload size limits
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    self.client.table('option_chain_data').insert(batch).execute()
            
            logger.info(f"Stored {len(records)} option records for {chain_data['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing option chain: {e}")
            return False
    
    async def calculate_gamma_sequence(self, symbol: str, expiry_date: str) -> Dict:
        """Calculate and store gamma sequence data"""
        try:
            # Get recent option data for this expiry
            cutoff_time = datetime.now() - timedelta(days=Config.GAMMA_LOOKBACK_DAYS)
            
            response = self.client.table('option_chain_data').select('*').eq(
                'symbol', symbol
            ).eq('expiry_date', expiry_date).gte(
                'timestamp', cutoff_time.isoformat()
            ).order('timestamp', desc=False).execute()
            
            data = response.data
            if not data:
                logger.warning(f"No data found for gamma calculation: {symbol}")
                return {}
            
            df = pd.DataFrame(data)
            
            # Group by timestamp to get snapshots
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            timestamps = df['timestamp'].unique()
            
            # Calculate gamma metrics for each timestamp
            gamma_records = []
            for ts in timestamps:
                ts_data = df[df['timestamp'] == ts]
                
                # Calculate total gamma
                total_gamma = ts_data['gamma'].abs().sum()
                call_gamma = ts_data[ts_data['option_type'] == 'CE']['gamma'].abs().sum()
                put_gamma = ts_data[ts_data['option_type'] == 'PE']['gamma'].abs().sum()
                gamma_exposure = call_gamma - put_gamma
                
                # Calculate gamma by strike
                gamma_by_strike = {}
                for strike in ts_data['strike_price'].unique():
                    strike_gamma = ts_data[
                        ts_data['strike_price'] == strike
                    ]['gamma'].abs().sum()
                    gamma_by_strike[str(strike)] = float(strike_gamma)
                
                # Find gamma walls (highest concentrations)
                gamma_walls = sorted(
                    gamma_by_strike.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                gamma_record = {
                    'symbol': symbol,
                    'expiry_date': expiry_date,
                    'total_gamma': float(total_gamma),
                    'call_gamma': float(call_gamma),
                    'put_gamma': float(put_gamma),
                    'gamma_exposure': float(gamma_exposure),
                    'gamma_levels': gamma_by_strike,
                    'gamma_walls': dict(gamma_walls),
                    'timestamp': ts.isoformat()
                }
                
                gamma_records.append(gamma_record)
            
            # Store the latest gamma data
            if gamma_records:
                latest = gamma_records[-1]
                self.client.table('gamma_data').insert(latest).execute()
                logger.info(f"Calculated gamma for {symbol}: Exposure={latest['gamma_exposure']:.2f}")
                return latest
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating gamma sequence: {e}")
            return {}
    
    async def detect_volume_spikes(self, symbol: str, 
                                 lookback_hours: int = None) -> List[Dict]:
        """Detect volume spikes in options"""
        try:
            if lookback_hours is None:
                lookback_hours = Config.LOOKBACK_HOURS_VOLUME
            
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            response = self.client.table('option_chain_data').select('*').eq(
                'symbol', symbol
            ).gte('timestamp', cutoff_time.isoformat()).order(
                'timestamp', desc=True
            ).execute()
            
            data = response.data
            if not data or len(data) < 10:
                return []
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            spikes = []
            
            # Group by strike and option type
            grouped = df.groupby(['strike_price', 'option_type', 'expiry_date'])
            
            for (strike, opt_type, expiry), group in grouped:
                if len(group) >= 5:  # Need enough data points
                    group = group.sort_values('timestamp')
                    
                    # Get current and historical volumes
                    current_volume = group.iloc[-1]['volume']
                    historical_volumes = group.iloc[:-1]['volume'].values
                    
                    if len(historical_volumes) >= 4:
                        avg_volume = np.mean(historical_volumes[-4:])  # Last 4 periods
                        std_volume = np.std(historical_volumes[-4:])
                        
                        if avg_volume > 0 and std_volume > 0:
                            volume_ratio = current_volume / avg_volume
                            z_score = (current_volume - avg_volume) / std_volume
                            
                            # Spike detection: >3x volume or z-score > 3
                            is_spike = (volume_ratio > 3.0) or (z_score > 3.0)
                            spike_score = max(volume_ratio, z_score)
                            
                            if is_spike:
                                spike_record = {
                                    'symbol': symbol,
                                    'strike_price': float(strike),
                                    'expiry_date': expiry,
                                    'option_type': opt_type,
                                    'current_volume': int(current_volume),
                                    'avg_volume': int(avg_volume),
                                    'std_volume': float(std_volume),
                                    'volume_ratio': float(volume_ratio),
                                    'z_score': float(z_score),
                                    'spike_score': float(spike_score),
                                    'is_spike': True,
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                self.client.table('volume_spikes').insert(spike_record).execute()
                                spikes.append(spike_record)
            
            if spikes:
                logger.info(f"Detected {len(spikes)} volume spikes for {symbol}")
            
            return spikes[:Config.MAX_VOLUME_SPIKES]
            
        except Exception as e:
            logger.error(f"Error detecting volume spikes: {e}")
            return []
    
    async def store_ml_prediction(self, prediction_data: Dict) -> bool:
        """Store ML prediction result"""
        try:
            record = {
                'symbol': prediction_data.get('symbol'),
                'expiry_date': prediction_data.get('expiry_date'),
                'prediction_type': prediction_data.get('prediction_type', 'expiry_spike'),
                'prediction_value': float(prediction_data.get('prediction_value', 0)),
                'confidence': float(prediction_data.get('confidence', 0)),
                'features': prediction_data.get('features', {}),
                'model_version': prediction_data.get('model_version', '1.0'),
                'trigger_time': datetime.now().isoformat()
            }
            
            self.client.table('ml_predictions').insert(record).execute()
            return True
            
        except Exception as e:
            logger.error(f"Error storing ML prediction: {e}")
            return False
    
    async def store_signal(self, signal_data: Dict) -> str:
        """Store trading signal and return signal ID"""
        try:
            record = {
                'signal_type': signal_data.get('signal_type', 'HOLD'),
                'symbol': signal_data.get('symbol'),
                'strike_price': signal_data.get('strike_price', 0),
                'expiry_date': signal_data.get('expiry_date'),
                'option_type': signal_data.get('option_type', ''),
                'action': signal_data.get('action', 'HOLD'),
                'price_target': float(signal_data.get('price_target', 0)),
                'stop_loss': float(signal_data.get('stop_loss', 0)),
                'rationale': signal_data.get('rationale', ''),
                'news_context': signal_data.get('news_context', []),
                'ai_model': signal_data.get('ai_model', 'Groq AI'),
                'analysis_time': float(signal_data.get('analysis_time_seconds', 0)),
                'sent': True,
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table('telegram_signals').insert(record).execute()
            
            if response.data and len(response.data) > 0:
                signal_id = response.data[0]['id']
                logger.info(f"Stored signal {signal_id} for {record['symbol']}")
                return signal_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            return None
    
    async def store_news(self, news_items: List[Dict]) -> bool:
        """Store news data"""
        try:
            records = []
            for news in news_items:
                record = {
                    'symbol': news.get('symbol'),
                    'title': news.get('title', ''),
                    'description': news.get('description', ''),
                    'content': news.get('content', ''),
                    'source': news.get('source', ''),
                    'published_at': news.get('published_at'),
                    'sentiment_score': float(news.get('sentiment', {}).get('score', 0)),
                    'sentiment_category': news.get('sentiment', {}).get('category', 'neutral'),
                    'keywords': news.get('keywords_found', []),
                    'relevance_score': float(news.get('relevance_score', 0)),
                    'url': news.get('url', ''),
                    'crawled_at': datetime.now().isoformat()
                }
                records.append(record)
            
            if records:
                self.client.table('news_data').insert(records).execute()
                logger.info(f"Stored {len(records)} news items")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error storing news: {e}")
            return False
    
    async def get_recent_signals(self, symbol: str = None, 
                               limit: int = 10) -> List[Dict]:
        """Get recent trading signals"""
        try:
            query = self.client.table('telegram_signals').select('*').order(
                'created_at', desc=True
            ).limit(limit)
            
            if symbol:
                query = query.eq('symbol', symbol)
            
            response = query.execute()
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    async def get_historical_gamma(self, symbol: str, 
                                 days_back: int = 7) -> pd.DataFrame:
        """Get historical gamma data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            response = self.client.table('gamma_data').select('*').eq(
                'symbol', symbol
            ).gte('timestamp', cutoff_time.isoformat()).order(
                'timestamp', desc=False
            ).execute()
            
            data = response.data
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical gamma: {e}")
            return pd.DataFrame()
    
    async def get_option_chain_history(self, symbol: str, expiry_date: str,
                                     hours_back: int = 24) -> pd.DataFrame:
        """Get option chain history for analysis"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            response = self.client.table('option_chain_data').select('*').eq(
                'symbol', symbol
            ).eq('expiry_date', expiry_date).gte(
                'timestamp', cutoff_time.isoformat()
            ).order('timestamp', desc=False).execute()
            
            data = response.data
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting option chain history: {e}")
            return pd.DataFrame()
    
    async def log_system_event(self, level: str, module: str, 
                             message: str, details: Dict = None):
        """Log system event to database"""
        try:
            record = {
                'level': level,
                'module': module,
                'message': message,
                'details': details or {},
                'timestamp': datetime.now().isoformat()
            }
            
            self.client.table('system_logs').insert(record).execute()
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to prevent database bloat"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            # Delete old option chain data
            self.client.table('option_chain_data').lt(
                'created_at', cutoff_str
            ).delete().execute()
            
            # Delete old volume spikes
            self.client.table('volume_spikes').lt(
                'created_at', cutoff_str
            ).delete().execute()
            
            # Delete old system logs
            self.client.table('system_logs').lt(
                'created_at', cutoff_str
            ).delete().execute()
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
