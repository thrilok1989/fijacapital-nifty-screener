# app.py (Simplified Streamlit App)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

st.set_page_config(
    page_title="Options Spike Detector",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
    }
    .signal-buy {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .signal-sell {
        background-color: #F44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📈 Options Expiry Spike Detector")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        symbol = st.selectbox(
            "Select Symbol",
            ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"],
            index=0
        )
        
        expiry_date = st.date_input(
            "Expiry Date",
            datetime.now() + timedelta(days=7)
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Gamma", "Volume", "OI", "IV"]
        )
        
        if st.button("Run Analysis", type="primary"):
            st.session_state.run_analysis = True
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Market Data - {symbol}")
        
        # Placeholder for real data
        if st.session_state.get('run_analysis', False):
            # Generate sample data
            dates = pd.date_range(end=datetime.now(), periods=50, freq='H')
            prices = np.random.normal(100, 5, 50).cumsum() + 10000
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
            fig.update_layout(title=f"{symbol} Price Trend", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                st.metric("Current Price", "₹18,250.50", "+125.75")
            with col2a:
                st.metric("Gamma Exposure", "₹1.2M", "-0.5M")
            with col3a:
                st.metric("PCR", "0.85", "-0.05")
            
            # Volume spike detection
            st.subheader("Volume Spike Analysis")
            spike_data = pd.DataFrame({
                'Strike': [18000, 18200, 18400, 18600, 18800],
                'Call Volume': [5000, 7500, 12000, 8000, 4000],
                'Put Volume': [3000, 4500, 6000, 9000, 5500],
                'Spike': [False, False, True, False, False]
            })
            
            st.dataframe(spike_data)
            
            # Signal generation
            st.subheader("AI Trading Signal")
            
            signal_type = "BUY_CALL" if np.random.random() > 0.5 else "BUY_PUT"
            confidence = np.random.uniform(0.6, 0.9)
            
            if signal_type == "BUY_CALL":
                st.markdown(f"""
                <div class="signal-buy">
                    <h3>📈 BUY CALL SIGNAL</h3>
                    <p>Confidence: {confidence:.1%}</p>
                    <p>Target: ₹18,500</p>
                    <p>Stop Loss: ₹18,100</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="signal-sell">
                    <h3>📉 BUY PUT SIGNAL</h3>
                    <p>Confidence: {confidence:.1%}</p>
                    <p>Target: ₹18,000</p>
                    <p>Stop Loss: ₹18,400</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Quick Stats")
        
        # Real-time metrics
        metrics = {
            "Spike Probability": "65.3%",
            "Max Pain": "₹18,250",
            "IV Percentile": "45.2%",
            "OI Concentration": "72.1%"
        }
        
        for key, value in metrics.items():
            st.metric(key, value)
        
        st.markdown("---")
        
        # News section
        st.subheader("Market News")
        
        news_items = [
            {"title": "FIIs increase derivative positions", "sentiment": "Positive"},
            {"title": "Expiry day volatility expected", "sentiment": "Neutral"},
            {"title": "Bank Nifty shows strong put writing", "sentiment": "Positive"}
        ]
        
        for news in news_items:
            st.write(f"📰 **{news['title']}**")
            st.caption(f"Sentiment: {news['sentiment']}")
            st.write("")

if __name__ == "__main__":
    main()
