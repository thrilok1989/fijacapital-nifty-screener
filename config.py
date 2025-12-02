import os
from datetime import datetime, time
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class Config:
    # Dhan API
    DHAN_CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
    DHAN_ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')
    
    # Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # Groq AI
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # News API
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    
    # ML Model
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/spike_detector.pkl')
    
    # System
    SYMBOLS = os.getenv('SYMBOLS', 'NIFTY,BANKNIFTY').split(',')
    CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '300'))
    SPIKE_THRESHOLD = float(os.getenv('SPIKE_THRESHOLD', '0.7'))
    ANALYSIS_CONFIDENCE_THRESHOLD = float(os.getenv('ANALYSIS_CONFIDENCE_THRESHOLD', '0.3'))
    MAX_VOLUME_SPIKES = int(os.getenv('MAX_VOLUME_SPIKES', '10'))
    MAX_NEWS_ITEMS = int(os.getenv('MAX_NEWS_ITEMS', '5'))
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_system.log')
    
    # Trading Hours (IST)
    MARKET_OPEN_TIME = time(
        int(os.getenv('MARKET_OPEN_HOUR', '9')),
        int(os.getenv('MARKET_OPEN_MINUTE', '15'))
    )
    MARKET_CLOSE_TIME = time(
        int(os.getenv('MARKET_CLOSE_HOUR', '15')),
        int(os.getenv('MARKET_CLOSE_MINUTE', '30'))
    )
    
    # Feature Engineering
    FEATURE_WINDOW_DAYS = 5
    LOOKBACK_HOURS_VOLUME = 24
    GAMMA_LOOKBACK_DAYS = 3
    
    # AI Settings
    AI_TEMPERATURE = 0.1
    AI_MAX_TOKENS = 4096
    AI_RESPONSE_FORMAT = "json_object"
    
    # Risk Management
    MAX_POSITION_SIZE_PERCENT = 2.5
    DEFAULT_STOP_LOSS_PERCENT = 1.0
    DEFAULT_TARGET_PERCENT = 2.0
    
    @classmethod
    def is_market_open(cls) -> bool:
        """Check if market is open based on IST time"""
        now = datetime.now().time()
        return cls.MARKET_OPEN_TIME <= now <= cls.MARKET_CLOSE_TIME
    
    @classmethod
    def get_trading_symbols(cls) -> List[str]:
        """Get list of trading symbols"""
        return cls.SYMBOLS
    
    @classmethod
    def get_groq_models(cls) -> Dict[str, str]:
        """Get available Groq models"""
        return {
            'analysis': 'llama3-70b-8192',
            'quick': 'llama3-8b-8192',
            'reasoning': 'mixtral-8x7b-32768',
            'default': cls.GROQ_MODEL
        }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        required_vars = [
            ('DHAN_CLIENT_ID', 'Dhan Client ID'),
            ('DHAN_ACCESS_TOKEN', 'Dhan Access Token'),
            ('SUPABASE_URL', 'Supabase URL'),
            ('SUPABASE_KEY', 'Supabase Key'),
            ('GROQ_API_KEY', 'Groq API Key'),
            ('TELEGRAM_BOT_TOKEN', 'Telegram Bot Token'),
            ('TELEGRAM_CHAT_ID', 'Telegram Chat ID'),
        ]
        
        for var_name, var_display in required_vars:
            if not getattr(cls, var_name):
                errors.append(f"{var_display} is not set")
        
        if not cls.SYMBOLS:
            errors.append("No trading symbols configured")
        
        return errors

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/cache', exist_ok=True)
os.makedirs('data/historical', exist_ok=True)
