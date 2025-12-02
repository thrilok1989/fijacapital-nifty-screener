# streamlit_config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit-specific configuration
class StreamlitConfig:
    # Page settings
    PAGE_TITLE = "AI Trading System"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # Theme
    PRIMARY_COLOR = "#1E88E5"
    BACKGROUND_COLOR = "#f0f2f6"
    SECONDARY_BACKGROUND_COLOR = "#ffffff"
    TEXT_COLOR = "#262730"
    FONT = "sans serif"
    
    # Features
    ENABLE_TELEGRAM = True
    ENABLE_AI_CHAT = True
    ENABLE_REAL_TIME_UPDATES = True
    UPDATE_INTERVAL = 300  # seconds
    
    # Display settings
    MAX_SIGNALS_DISPLAY = 10
    MAX_NEWS_DISPLAY = 5
    CHAT_HISTORY_LIMIT = 20
    
    # Chart settings
    CHART_HEIGHT = 400
    CHART_THEME = "plotly_white"
    
    @staticmethod
    def get_symbol_display_name(symbol: str) -> str:
        """Get display name for symbols"""
        display_names = {
            'NIFTY': 'Nifty 50',
            'BANKNIFTY': 'Bank Nifty',
            'FINNIFTY': 'Fin Nifty'
        }
        return display_names.get(symbol, symbol)
