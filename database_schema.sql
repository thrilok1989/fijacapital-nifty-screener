-- Run this in Supabase SQL Editor

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Options chain data
CREATE TABLE IF NOT EXISTS option_chain_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    expiry_date DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CE', 'PE')),
    ltp DECIMAL(10,2),
    volume INTEGER,
    oi INTEGER,
    change_in_oi INTEGER,
    iv DECIMAL(5,2),
    delta DECIMAL(5,4),
    gamma DECIMAL(5,4),
    theta DECIMAL(5,4),
    vega DECIMAL(5,4),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, strike_price, expiry_date, option_type, timestamp)
);

-- Gamma sequence data
CREATE TABLE IF NOT EXISTS gamma_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    expiry_date DATE NOT NULL,
    total_gamma DECIMAL(15,2),
    call_gamma DECIMAL(15,2),
    put_gamma DECIMAL(15,2),
    gamma_exposure DECIMAL(15,2),
    gamma_levels JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, expiry_date, timestamp)
);

-- Volume spike detection
CREATE TABLE IF NOT EXISTS volume_spikes (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strike_price DECIMAL(10,2),
    expiry_date DATE,
    option_type VARCHAR(4) CHECK (option_type IN ('CE', 'PE')),
    current_volume INTEGER,
    avg_volume INTEGER,
    volume_ratio DECIMAL(10,2),
    spike_score DECIMAL(10,2),
    is_spike BOOLEAN,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_volume_spikes_symbol_timestamp 
ON volume_spikes(symbol, timestamp DESC);

-- ML predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    expiry_date DATE,
    prediction_type VARCHAR(50),
    prediction_value DECIMAL(10,4),
    confidence DECIMAL(5,4),
    features JSONB,
    model_version VARCHAR(50),
    trigger_time TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Telegram signals
CREATE TABLE IF NOT EXISTS telegram_signals (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    signal_type VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strike_price DECIMAL(10,2),
    expiry_date DATE,
    option_type VARCHAR(4),
    action VARCHAR(20),
    price_target DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    rationale TEXT,
    news_context JSONB,
    ai_model VARCHAR(50),
    analysis_time DECIMAL(10,2),
    sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- News data
CREATE TABLE IF NOT EXISTS news_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20),
    title TEXT,
    description TEXT,
    content TEXT,
    source VARCHAR(100),
    published_at TIMESTAMPTZ,
    sentiment_score DECIMAL(5,4),
    sentiment_category VARCHAR(20),
    keywords JSONB,
    relevance_score DECIMAL(5,4),
    url TEXT,
    crawled_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trading performance
CREATE TABLE IF NOT EXISTS trading_performance (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    signal_id UUID REFERENCES telegram_signals(id),
    entry_price DECIMAL(10,2),
    exit_price DECIMAL(10,2),
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    pnl DECIMAL(10,2),
    pnl_percent DECIMAL(5,2),
    status VARCHAR(20) CHECK (status IN ('OPEN', 'CLOSED', 'STOPPED', 'TARGET_HIT')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    module VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_option_chain_symbol_expiry 
ON option_chain_data(symbol, expiry_date, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_time 
ON ml_predictions(symbol, trigger_time DESC);

CREATE INDEX IF NOT EXISTS idx_telegram_signals_created 
ON telegram_signals(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_news_data_symbol_published 
ON news_data(symbol, published_at DESC);

-- Create materialized view for daily summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_summary AS
SELECT 
    DATE(timestamp) as trade_date,
    symbol,
    COUNT(DISTINCT expiry_date) as expiry_count,
    SUM(total_volume) as total_volume,
    AVG(gamma_exposure) as avg_gamma_exposure,
    COUNT(*) as record_count
FROM (
    SELECT 
        oc.symbol,
        oc.expiry_date,
        oc.timestamp,
        SUM(oc.volume) as total_volume,
        gd.gamma_exposure
    FROM option_chain_data oc
    LEFT JOIN gamma_data gd ON oc.symbol = gd.symbol 
        AND oc.expiry_date = gd.expiry_date 
        AND DATE(oc.timestamp) = DATE(gd.timestamp)
    GROUP BY oc.symbol, oc.expiry_date, oc.timestamp, gd.gamma_exposure
) aggregated
GROUP BY DATE(timestamp), symbol
ORDER BY trade_date DESC, symbol;

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW daily_summary;
