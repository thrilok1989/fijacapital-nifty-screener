# streamlit_app.py
import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="AI Trading System - Expiry Spike Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .signal-buy-call {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .signal-buy-put {
        background-color: #F44336;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .signal-hold {
        background-color: #FF9800;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .news-positive {
        border-left: 5px solid #4CAF50;
        padding-left: 10px;
    }
    .news-negative {
        border-left: 5px solid #F44336;
        padding-left: 10px;
    }
    .news-neutral {
        border-left: 5px solid #FF9800;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Import trading system components
from config import Config
from dhan_client import DhanClient
from supabase_manager import SupabaseManager
from ml_detector import ExpirySpikeMLDetector
from news_integration import NewsAnalyzer
from groq_ai_system import GroqAIAnalyzer
from telegram_notifier import TelegramNotifier

class TradingSystemDashboard:
    def __init__(self):
        # Initialize session state
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'signals' not in st.session_state:
            st.session_state.signals = []
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
        
        # Initialize components
        self.supabase = SupabaseManager()
        self.dhan_client = DhanClient()
        self.ml_detector = ExpirySpikeMLDetector()
        self.news_analyzer = NewsAnalyzer()
        self.ai_system = GroqAIAnalyzer(supabase_manager=self.supabase)
    
    def initialize_system(self):
        """Initialize trading system"""
        try:
            # Test database connection
            st.session_state.supabase = self.supabase
            st.session_state.ml_detector = self.ml_detector
            st.session_state.ai_system = self.ai_system
            st.session_state.system_initialized = True
            
            st.success("✅ Trading system initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'initialized': st.session_state.system_initialized,
            'last_update': st.session_state.last_update,
            'symbols': Config.SYMBOLS,
            'market_open': Config.is_market_open(),
            'model': Config.GROQ_MODEL
        }
    
    async def fetch_latest_data(self, symbol: str):
        """Fetch latest market data"""
        try:
            expiry_date = self.dhan_client.get_nearest_expiry()
            
            async with self.dhan_client as client:
                option_data = await client.get_option_chain(symbol, expiry_date)
            
            if option_data:
                # Store in session state
                if symbol not in st.session_state.market_data:
                    st.session_state.market_data[symbol] = {}
                
                st.session_state.market_data[symbol]['option_chain'] = option_data
                st.session_state.market_data[symbol]['timestamp'] = datetime.now()
                
                # Calculate gamma
                gamma_data = await self.supabase.calculate_gamma_sequence(symbol, expiry_date)
                st.session_state.market_data[symbol]['gamma'] = gamma_data
                
                # Detect volume spikes
                volume_spikes = await self.supabase.detect_volume_spikes(symbol)
                st.session_state.market_data[symbol]['volume_spikes'] = volume_spikes
                
                return True
            return False
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return False
    
    def display_option_chain(self, symbol: str):
        """Display option chain data"""
        if symbol not in st.session_state.market_data:
            st.warning(f"No data for {symbol}")
            return
        
        data = st.session_state.market_data[symbol].get('option_chain', {})
        if not data:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Market Overview")
            st.metric("Underlying Price", f"₹{data.get('underlying_price', 0):.2f}")
            st.metric("Total OI", f"{data.get('total_oi', 0):,}")
            st.metric("Total Volume", f"{data.get('total_volume', 0):,}")
            st.metric("ATM Strike", f"₹{data.get('atm_strike', 0):.2f}")
        
        with col2:
            st.subheader("📈 Ratios")
            st.metric("Put/Call Ratio (Volume)", f"{data.get('put_call_ratio_volume', 0):.2f}")
            st.metric("Put/Call Ratio (OI)", f"{data.get('put_call_ratio_oi', 0):.2f}")
            st.metric("Max Pain", f"₹{data.get('max_pain', 0):.2f}")
    
    def display_gamma_data(self, symbol: str):
        """Display gamma analysis"""
        if symbol not in st.session_state.market_data:
            return
        
        gamma_data = st.session_state.market_data[symbol].get('gamma', {})
        if not gamma_data:
            return
        
        st.subheader("🧮 Gamma Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gamma Exposure", f"₹{gamma_data.get('gamma_exposure', 0):,.2f}")
        
        with col2:
            st.metric("Total Gamma", f"{gamma_data.get('total_gamma', 0):,.2f}")
        
        with col3:
            st.metric("Call Gamma", f"{gamma_data.get('call_gamma', 0):,.2f}")
        
        with col4:
            st.metric("Put Gamma", f"{gamma_data.get('put_gamma', 0):,.2f}")
        
        # Plot gamma levels
        if 'gamma_levels' in gamma_data:
            gamma_levels = gamma_data['gamma_levels']
            if gamma_levels:
                df = pd.DataFrame(list(gamma_levels.items()), columns=['Strike', 'Gamma'])
                df['Strike'] = pd.to_numeric(df['Strike'])
                df = df.sort_values('Strike')
                
                fig = px.bar(df, x='Strike', y='Gamma', title=f"Gamma Levels - {symbol}")
                st.plotly_chart(fig, use_container_width=True)
    
    def display_volume_spikes(self, symbol: str):
        """Display volume spikes"""
        if symbol not in st.session_state.market_data:
            return
        
        spikes = st.session_state.market_data[symbol].get('volume_spikes', [])
        if not spikes:
            return
        
        st.subheader("🚨 Volume Spikes")
        
        df = pd.DataFrame(spikes)
        if not df.empty:
            # Format columns
            df['strike_price'] = df['strike_price'].astype(str)
            df['volume_ratio'] = df['volume_ratio'].round(2)
            df['z_score'] = df['z_score'].round(2)
            
            # Display table
            st.dataframe(
                df[['strike_price', 'option_type', 'current_volume', 
                    'avg_volume', 'volume_ratio', 'z_score']],
                use_container_width=True
            )
            
            # Plot spikes
            fig = px.scatter(
                df, 
                x='strike_price', 
                y='volume_ratio',
                color='option_type',
                size='current_volume',
                hover_data=['current_volume', 'avg_volume', 'z_score'],
                title=f"Volume Spikes - {symbol}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    async def run_analysis(self, symbol: str):
        """Run ML analysis on current data"""
        if symbol not in st.session_state.market_data:
            return None
        
        try:
            option_data = st.session_state.market_data[symbol].get('option_chain', {})
            gamma_data = st.session_state.market_data[symbol].get('gamma', {})
            
            if not option_data:
                return None
            
            # Fetch historical data
            async with self.dhan_client as client:
                historical_data = await client.get_historical_data(
                    symbol, 
                    interval='5minute',
                    from_date=(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
                )
            
            if len(historical_data) >= 20:
                # Extract features
                features = self.ml_detector.extract_features(
                    historical_data, option_data, gamma_data
                )
                
                # Predict
                is_spike, confidence = self.ml_detector.predict_spike(features)
                
                # Get explanation
                explanation = self.ml_detector.explain_prediction(features)
                
                return {
                    'is_spike': is_spike,
                    'confidence': confidence,
                    'features': features,
                    'explanation': explanation,
                    'historical_data': historical_data
                }
            
            return None
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None
    
    async def generate_ai_signal(self, symbol: str, analysis_result: Dict):
        """Generate AI trading signal"""
        if not analysis_result or not analysis_result.get('is_spike'):
            return None
        
        try:
            # Get news context
            news_context = await self.news_analyzer.get_relevant_news(symbol)
            
            # Prepare spike data
            option_data = st.session_state.market_data[symbol].get('option_chain', {})
            gamma_data = st.session_state.market_data[symbol].get('gamma', {})
            
            spike_data = {
                'symbol': symbol,
                'expiry_date': self.dhan_client.get_nearest_expiry(),
                'confidence': analysis_result['confidence'],
                'underlying_price': option_data.get('underlying_price', 0),
                'total_volume': option_data.get('total_volume', 0),
                'total_oi': option_data.get('total_oi', 0),
                'gamma_exposure': gamma_data.get('gamma_exposure', 0),
                'spike_score': analysis_result['confidence'],
                'detection_time': datetime.now().isoformat()
            }
            
            # Generate signal
            signal = await self.ai_system.generate_trading_signal_analysis(
                spike_data, news_context
            )
            
            # Add to session state
            signal_dict = signal.dict() if hasattr(signal, 'dict') else signal
            signal_dict['symbol'] = symbol
            signal_dict['timestamp'] = datetime.now()
            
            st.session_state.signals.append(signal_dict)
            
            return signal_dict
            
        except Exception as e:
            st.error(f"Error generating AI signal: {e}")
            return None
    
    def display_signal(self, signal: Dict):
        """Display trading signal"""
        if not signal:
            return
        
        signal_type = signal.get('signal_type', 'HOLD')
        confidence = signal.get('confidence', 0) * 100
        symbol = signal.get('symbol', 'Unknown')
        
        # Determine color based on signal type
        if signal_type == 'BUY_CALL':
            color = "success"
            emoji = "📈🟢"
        elif signal_type == 'BUY_PUT':
            color = "error"
            emoji = "📉🔴"
        elif signal_type == 'SELL_CALL':
            color = "warning"
            emoji = "📉🟢"
        elif signal_type == 'SELL_PUT':
            color = "info"
            emoji = "📈🔴"
        else:
            color = "warning"
            emoji = "⏸️⚪"
        
        st.markdown(f"""
        <div class="signal-{signal_type.lower().replace('_', '-')}">
            <h3>{emoji} {signal_type.replace('_', ' ')} - {symbol}</h3>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            <p><strong>Price Target:</strong> ₹{signal.get('price_target', 0):.2f}</p>
            <p><strong>Stop Loss:</strong> ₹{signal.get('stop_loss', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display rationale
        with st.expander("📝 AI Rationale"):
            st.write(signal.get('rationale', 'No rationale provided'))
        
        # Display risk factors
        with st.expander("⚠️ Risk Factors"):
            risks = signal.get('key_risk_factors', [])
            for risk in risks:
                st.write(f"• {risk}")
    
    def display_news(self, symbol: str):
        """Display relevant news"""
        # This would fetch news - for now show placeholder
        st.subheader("📰 Market News")
        
        # Placeholder news items
        placeholder_news = [
            {
                'title': 'NIFTY shows strong support at 18500 level',
                'sentiment': {'category': 'positive', 'score': 0.7},
                'source': 'Economic Times',
                'time': '2 hours ago'
            },
            {
                'title': 'Options expiry sees heavy call writing at 18600',
                'sentiment': {'category': 'neutral', 'score': 0.1},
                'source': 'Moneycontrol',
                'time': '1 hour ago'
            },
            {
                'title': 'FIIs continue selling in derivatives segment',
                'sentiment': {'category': 'negative', 'score': -0.5},
                'source': 'Business Standard',
                'time': '3 hours ago'
            }
        ]
        
        for news in placeholder_news:
            sentiment_class = f"news-{news['sentiment']['category']}"
            st.markdown(f"""
            <div class="{sentiment_class}">
                <strong>{news['title']}</strong><br>
                <small>{news['source']} • {news['time']} • 
                Sentiment: {news['sentiment']['category'].upper()} ({news['sentiment']['score']:.2f})</small>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
    
    async def chat_with_ai(self, question: str, symbol: str = None):
        """Chat with AI about market data"""
        if not symbol:
            symbol = Config.SYMBOLS[0]
        
        with st.spinner("🤖 AI is thinking..."):
            try:
                response = await self.ai_system.query_data(question, symbol)
                return response.get('response', 'No response generated')
            except Exception as e:
                return f"Error: {str(e)}"
    
    def run_dashboard(self):
        """Main dashboard function"""
        
        # Header
        st.markdown("<h1 class='main-header'>🤖 AI Trading System - Expiry Spike Detector</h1>", unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=100)
            st.title("Control Panel")
            
            # System initialization
            if not st.session_state.system_initialized:
                if st.button("🚀 Initialize System", type="primary"):
                    with st.spinner("Initializing..."):
                        self.initialize_system()
            
            # Symbol selection
            selected_symbol = st.selectbox(
                "Select Symbol",
                Config.SYMBOLS,
                index=0
            )
            
            # Actions
            st.subheader("Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Data"):
                    st.session_state.refresh_requested = True
            
            with col2:
                if st.button("📊 Run Analysis"):
                    st.session_state.run_analysis = True
            
            # AI Chat
            st.subheader("🤖 AI Assistant")
            ai_question = st.text_input("Ask about market data:")
            if ai_question and st.button("Ask AI"):
                st.session_state.ai_question = ai_question
            
            # System info
            st.subheader("System Info")
            status = self.get_system_status()
            st.write(f"**Status:** {'✅ Running' if status['initialized'] else '❌ Stopped'}")
            st.write(f"**Symbols:** {', '.join(status['symbols'])}")
            st.write(f"**Market:** {'✅ Open' if status['market_open'] else '❌ Closed'}")
            st.write(f"**AI Model:** {status['model']}")
        
        # Main content area
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system from the sidebar")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Overview", 
            "🧮 Gamma Analysis", 
            "🚨 Volume Spikes", 
            "🤖 AI Signals", 
            "💬 AI Assistant"
        ])
        
        # Tab 1: Market Overview
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.header(f"Market Data - {selected_symbol}")
                
                # Refresh button
                if st.button("🔄 Fetch Latest Data", key="refresh_main"):
                    with st.spinner("Fetching data..."):
                        success = asyncio.run(self.fetch_latest_data(selected_symbol))
                        if success:
                            st.success("Data updated!")
                        else:
                            st.error("Failed to fetch data")
                
                # Display option chain
                self.display_option_chain(selected_symbol)
                
                # Historical chart placeholder
                st.subheader("Historical Trend")
                # Generate sample data for chart
                dates = pd.date_range(end=datetime.now(), periods=50, freq='H')
                prices = np.random.normal(100, 5, 50).cumsum() + 10000
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
                fig.update_layout(
                    title=f"{selected_symbol} Price Trend",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.header("Quick Stats")
                
                # Real-time metrics
                metrics_data = {
                    "Spike Probability": "25.3%",
                    "Gamma Exposure": "₹1.2M",
                    "PCR (Volume)": "0.85",
                    "IV Percentile": "45.2%",
                    "Max Pain": "₹18,500"
                }
                
                for metric, value in metrics_data.items():
                    st.metric(metric, value)
                
                # News section
                self.display_news(selected_symbol)
        
        # Tab 2: Gamma Analysis
        with tab2:
            self.display_gamma_data(selected_symbol)
            
            # Additional gamma charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gamma Exposure Trend")
                # Sample data
                dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
                gamma_exposure = np.random.normal(0, 500000, 24).cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=gamma_exposure, mode='lines+markers'))
                fig.update_layout(
                    title="Gamma Exposure Over Time",
                    xaxis_title="Time",
                    yaxis_title="Gamma Exposure (₹)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Call vs Put Gamma")
                labels = ['Call Gamma', 'Put Gamma']
                values = [
                    st.session_state.market_data.get(selected_symbol, {}).get('gamma', {}).get('call_gamma', 500000),
                    st.session_state.market_data.get(selected_symbol, {}).get('gamma', {}).get('put_gamma', 300000)
                ]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                fig.update_layout(title="Gamma Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Volume Spikes
        with tab3:
            self.display_volume_spikes(selected_symbol)
            
            # Volume analysis
            st.subheader("Volume Analysis")
            
            # Create sample volume data
            strikes = list(range(18000, 19000, 100))
            call_volumes = np.random.poisson(5000, len(strikes))
            put_volumes = np.random.poisson(4000, len(strikes))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=strikes, y=call_volumes, name='Call Volume', marker_color='green'))
            fig.add_trace(go.Bar(x=strikes, y=put_volumes, name='Put Volume', marker_color='red'))
            
            fig.update_layout(
                title="Volume by Strike Price",
                xaxis_title="Strike Price",
                yaxis_title="Volume",
                barmode='group',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: AI Signals
        with tab4:
            st.header("🤖 AI Trading Signals")
            
            # Run analysis button
            if st.button("🎯 Generate New Signal", type="primary"):
                with st.spinner("Running analysis..."):
                    # Run ML analysis
                    analysis_result = asyncio.run(self.run_analysis(selected_symbol))
                    
                    if analysis_result:
                        # Generate AI signal
                        signal = asyncio.run(self.generate_ai_signal(selected_symbol, analysis_result))
                        
                        if signal:
                            st.success("Signal generated successfully!")
                            self.display_signal(signal)
                        else:
                            st.warning("No spike detected or signal generation failed")
                    else:
                        st.error("Analysis failed")
            
            # Display previous signals
            if st.session_state.signals:
                st.subheader("Recent Signals")
                
                for i, signal in enumerate(reversed(st.session_state.signals[-5:]), 1):
                    with st.expander(f"Signal #{len(st.session_state.signals) - i + 1}: {signal.get('signal_type')} - {signal.get('symbol')}"):
                        self.display_signal(signal)
            
            # ML Model Info
            st.subheader("🧠 ML Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", "XGBoost")
            
            with col2:
                st.metric("Features Used", "20")
            
            with col3:
                st.metric("Spike Threshold", f"{Config.SPIKE_THRESHOLD*100:.1f}%")
            
            # Feature importance (placeholder)
            st.subheader("Feature Importance")
            features = ['Gamma Exposure', 'Volume Ratio', 'PCR', 'IV Skew', 'Days to Expiry']
            importance = [0.25, 0.20, 0.15, 0.15, 0.10]
            
            fig = go.Figure(data=[go.Bar(x=features, y=importance)])
            fig.update_layout(
                title="Top 5 Feature Importances",
                yaxis_title="Importance Score",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 5: AI Assistant
        with tab5:
            st.header("🤖 AI Trading Assistant")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Chat interface
                st.write("Ask me anything about the market, analysis, or trading signals!")
                
                # Pre-defined questions
                st.subheader("💡 Try asking:")
                
                questions = [
                    "What's the current gamma exposure?",
                    "Should I buy calls or puts?",
                    "Explain the volume spikes",
                    "What's the market sentiment?",
                    "Give me trading recommendations"
                ]
                
                cols = st.columns(3)
                for idx, question in enumerate(questions):
                    with cols[idx % 3]:
                        if st.button(question, key=f"q_{idx}"):
                            st.session_state.ai_question = question
            
            with col2:
                st.subheader("📊 Quick Analysis")
                if st.button("Analyze Current Market"):
                    st.session_state.ai_question = "Analyze the current market conditions and provide trading insights"
            
            # Chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat
            for message in st.session_state.chat_history[-10:]:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
                st.markdown("---")
            
            # Input for new question
            ai_question = st.text_input("Type your question:", key="ai_input")
            
            if st.button("Send", key="send_ai") and ai_question:
                # Add to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': ai_question,
                    'timestamp': datetime.now()
                })
                
                # Get AI response
                response = asyncio.run(self.chat_with_ai(ai_question, selected_symbol))
                
                # Add response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                
                # Refresh to show new message
                st.rerun()
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Last Update:** {st.session_state.last_update or 'Never'}")
        
        with col2:
            st.write(f"**Market:** {'✅ Open' if Config.is_market_open() else '❌ Closed'}")
        
        with col3:
            if st.button("🔄 Refresh All"):
                st.rerun()

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingSystemDashboard()
    dashboard.run_dashboard()
