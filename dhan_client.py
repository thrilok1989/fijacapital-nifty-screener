import os
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from config import Config

logger = logging.getLogger(__name__)

class DhanClient:
    """Async Dhan API Client with enhanced features"""
    
    BASE_URL = "https://api.dhan.co"
    
    def __init__(self, client_id: str = None, access_token: str = None):
        self.client_id = client_id or Config.DHAN_CLIENT_ID
        self.access_token = access_token or Config.DHAN_ACCESS_TOKEN
        
        if not self.client_id or not self.access_token:
            raise ValueError("Dhan Client ID and Access Token are required")
        
        self.headers = {
            'access-token': self.access_token,
            'Content-Type': 'application/json'
        }
        
        self.session = None
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key"""
        return f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Dict = None, data: Dict = None) -> Dict:
        """Make async HTTP request with caching"""
        cache_key = self._get_cache_key(endpoint, params or {})
        
        # Check cache
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cache_data
        
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params) as response:
                    result = await response.json()
            elif method.upper() == 'POST':
                async with self.session.post(url, json=data) as response:
                    result = await response.json()
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Cache successful responses
            if response.status == 200:
                self.cache[cache_key] = (time.time(), result)
            
            return result
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def get_option_chain(self, symbol: str, expiry_date: str) -> Dict:
        """Get option chain data for a symbol and expiry"""
        try:
            endpoint = f"/options/{symbol}"
            params = {'expiry': expiry_date}
            
            data = await self._make_request('GET', endpoint, params=params)
            
            if not data:
                return {}
            
            return self._process_option_chain(data, symbol, expiry_date)
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return {}
    
    def _process_option_chain(self, data: Dict, symbol: str, expiry_date: str) -> Dict:
        """Process raw option chain data"""
        processed = {
            'symbol': symbol,
            'expiry_date': expiry_date,
            'timestamp': datetime.now().isoformat(),
            'underlying_price': data.get('underlyingPrice', 0),
            'total_oi': data.get('totalOI', 0),
            'total_volume': data.get('totalVolume', 0),
            'atm_strike': self._calculate_atm_strike(
                data.get('underlyingPrice', 0),
                data.get('callOptions', []),
                data.get('putOptions', [])
            ),
            'calls': [],
            'puts': []
        }
        
        # Process calls
        for call in data.get('callOptions', []):
            processed_call = {
                'strike': call.get('strikePrice', 0),
                'ltp': call.get('lastPrice', 0),
                'volume': call.get('volume', 0),
                'oi': call.get('openInterest', 0),
                'change_in_oi': call.get('changeinOpenInterest', 0),
                'iv': call.get('impliedVolatility', 0),
                'delta': call.get('delta', 0),
                'gamma': call.get('gamma', 0),
                'theta': call.get('theta', 0),
                'vega': call.get('vega', 0),
                'bid': call.get('bid', 0),
                'ask': call.get('ask', 0),
                'bid_qty': call.get('bidQty', 0),
                'ask_qty': call.get('askQty', 0)
            }
            processed['calls'].append(processed_call)
        
        # Process puts
        for put in data.get('putOptions', []):
            processed_put = {
                'strike': put.get('strikePrice', 0),
                'ltp': put.get('lastPrice', 0),
                'volume': put.get('volume', 0),
                'oi': put.get('openInterest', 0),
                'change_in_oi': put.get('changeinOpenInterest', 0),
                'iv': put.get('impliedVolatility', 0),
                'delta': put.get('delta', 0),
                'gamma': put.get('gamma', 0),
                'theta': put.get('theta', 0),
                'vega': put.get('vega', 0),
                'bid': put.get('bid', 0),
                'ask': put.get('ask', 0),
                'bid_qty': put.get('bidQty', 0),
                'ask_qty': put.get('askQty', 0)
            }
            processed['puts'].append(processed_put)
        
        # Calculate additional metrics
        processed.update(self._calculate_chain_metrics(processed))
        
        return processed
    
    def _calculate_atm_strike(self, underlying_price: float, 
                             calls: List, puts: List) -> float:
        """Calculate ATM strike price"""
        if not calls and not puts:
            return underlying_price
        
        all_strikes = set()
        all_strikes.update([c.get('strikePrice', 0) for c in calls])
        all_strikes.update([p.get('strikePrice', 0) for p in puts])
        
        if not all_strikes:
            return underlying_price
        
        # Find strike closest to underlying price
        closest_strike = min(all_strikes, key=lambda x: abs(x - underlying_price))
        return closest_strike
    
    def _calculate_chain_metrics(self, chain_data: Dict) -> Dict:
        """Calculate additional option chain metrics"""
        metrics = {
            'put_call_ratio_volume': 0,
            'put_call_ratio_oi': 0,
            'iv_percentile': 0,
            'max_pain': 0
        }
        
        total_call_volume = sum(c['volume'] for c in chain_data['calls'])
        total_put_volume = sum(p['volume'] for p in chain_data['puts'])
        
        total_call_oi = sum(c['oi'] for c in chain_data['calls'])
        total_put_oi = sum(p['oi'] for p in chain_data['puts'])
        
        if total_call_volume > 0:
            metrics['put_call_ratio_volume'] = total_put_volume / total_call_volume
        
        if total_call_oi > 0:
            metrics['put_call_ratio_oi'] = total_put_oi / total_call_oi
        
        # Calculate max pain
        if chain_data['calls'] and chain_data['puts']:
            metrics['max_pain'] = self._calculate_max_pain(
                chain_data['calls'], chain_data['puts']
            )
        
        return metrics
    
    def _calculate_max_pain(self, calls: List, puts: List) -> float:
        """Calculate max pain strike"""
        strike_pain = {}
        
        for call in calls:
            strike = call['strike']
            oi = call['oi']
            for s in strike_pain:
                if s < strike:
                    strike_pain[s] += oi * (strike - s)
        
        for put in puts:
            strike = put['strike']
            oi = put['oi']
            for s in strike_pain:
                if s > strike:
                    strike_pain[s] += oi * (s - strike)
        
        if not strike_pain:
            return 0
        
        return min(strike_pain.items(), key=lambda x: x[1])[0]
    
    async def get_historical_data(self, symbol: str, interval: str = '5minute',
                                 from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=Config.FEATURE_WINDOW_DAYS)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            
            endpoint = "/charts/historical"
            params = {
                'symbol': symbol,
                'exchange': 'NSE',
                'instrument': 'EQUITY',
                'interval': interval,
                'from_date': from_date,
                'to_date': to_date
            }
            
            data = await self._make_request('GET', endpoint, params=params)
            
            if not data or 'data' not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['data'])
            
            # Convert and enhance DataFrame
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Add additional calculated columns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_greeks(self, symbol: str, strike: float, 
                        expiry: str, option_type: str) -> Dict:
        """Get Greek values for specific option"""
        try:
            endpoint = "/options/greeks"
            params = {
                'symbol': symbol,
                'strike': strike,
                'expiry': expiry,
                'type': option_type
            }
            
            return await self._make_request('GET', endpoint, params=params)
            
        except Exception as e:
            logger.error(f"Error fetching Greeks: {e}")
            return {}
    
    async def get_multiple_expiries(self, symbol: str) -> List[str]:
        """Get available expiry dates for a symbol"""
        try:
            endpoint = f"/options/expiries/{symbol}"
            data = await self._make_request('GET', endpoint)
            return data.get('expiries', [])[:4]  # Next 4 expiries
            
        except Exception as e:
            logger.error(f"Error fetching expiries for {symbol}: {e}")
            return []
    
    async def get_underlying_quote(self, symbol: str) -> Dict:
        """Get underlying stock/index quote"""
        try:
            endpoint = f"/quotes/{symbol}"
            return await self._make_request('GET', endpoint)
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_nearest_expiry(self, symbol: str = None) -> str:
        """Get nearest Thursday expiry date"""
        today = datetime.now()
        
        # Find next Thursday
        days_ahead = 3 - today.weekday()  # 3 = Thursday
        if days_ahead <= 0:
            days_ahead += 7
        
        expiry_date = today + timedelta(days=days_ahead)
        return expiry_date.strftime('%Y-%m-%d')
    
    async def batch_fetch_option_chains(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch fetch option chains for multiple symbols"""
        results = {}
        expiry = self.get_nearest_expiry()
        
        tasks = []
        for symbol in symbols:
            task = self.get_option_chain(symbol, expiry)
            tasks.append(task)
        
        # Run concurrently
        chain_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, chain_results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching chain for {symbol}: {result}")
                results[symbol] = {}
            else:
                results[symbol] = result
        
        return results
