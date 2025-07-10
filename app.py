import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Nifty Option Screener", layout="wide")

# 🔄 Auto refresh every 3 minutes
def auto_refresh(interval_sec=180):
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    if time.time() - st.session_state["last_refresh"] > interval_sec:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

auto_refresh(180)

st.title("📊 NIFTY Option Screener – Fijacapital")
st.markdown("⏰ Auto-refresh every 3 minutes | 🔄 Live NSE Option Chain Analysis")

@st.cache_data(ttl=600)
def fetch_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=headers)
        r = s.get(url, headers=headers, timeout=10)
        return r.json()
    except:
        st.warning("⚠️ Could not fetch data from NSE. Try again later.")
        return None

# 🔄 Fetch data
data = fetch_option_chain()
if data is None:
    st.stop()

raw_data = data['records']['data']
expiry_list = data['records']['expiryDates']
underlying = data['records'].get('underlyingValue', 0)

# 🗓️ Expiry Date Selection
selected_expiry = st.selectbox("📅 Select Expiry Date", expiry_list)

# Lists to hold CE and PE data
ce_list, pe_list = [], []

for item in raw_data:
    if item.get("expiryDate") == selected_expiry:
        strike = item.get("strikePrice", 0)

        if "CE" in item:
            ce = item["CE"]
            ce_list.append({
                "strikePrice": strike,
                "OI_CE": ce.get("openInterest", 0),
                "Chg_OI_CE": ce.get("changeinOpenInterest", 0),
                "Vol_CE": ce.get("totalTradedVolume", 0),
            })

        if "PE" in item:
            pe = item["PE"]
            pe_list.append({
                "strikePrice": strike,
                "OI_PE": pe.get("openInterest", 0),
                "Chg_OI_PE": pe.get("changeinOpenInterest", 0),
                "Vol_PE": pe.get("totalTradedVolume", 0),
            })

df_ce = pd.DataFrame(ce_list)
df_pe = pd.DataFrame(pe_list)

# 🎯 ATM Strike
atm_strike = min(df_ce['strikePrice'], key=lambda x: abs(x - underlying))

# Filter ±500 points around ATM
df_ce = df_ce[(df_ce["strikePrice"] >= atm_strike - 500) & (df_ce["strikePrice"] <= atm_strike + 500)]
df_pe = df_pe[(df_pe["strikePrice"] >= atm_strike - 500) & (df_pe["strikePrice"] <= atm_strike + 500)]

# 📉 PCR Calculation
total_ce_oi = df_ce["OI_CE"].sum()
total_pe_oi = df_pe["OI_PE"].sum()
pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

# 📊 PCR & TREND METER Display
col1, col2 = st.columns(2)
with col1:
    st.metric(label="📉 Put Call Ratio (PCR)", value=pcr)
with col2:
    if pcr > 1.2:
        st.success("🟢 Bullish Sentiment")
    elif pcr < 0.8:
        st.error("🔴 Bearish Sentiment")
    else:
        st.warning("🟡 Neutral Sentiment")

# Merge CE and PE
merged_df = pd.merge(df_ce, df_pe, on="strikePrice", how="outer").sort_values("strikePrice")

# 🧾 Display Table
st.markdown("### 🧾 Combined Option Chain (CALL + PUT Side-by-Side)")
st.caption(f"📍 Spot: `{underlying}` | 🎯 ATM Strike: `{atm_strike}` | 📅 Expiry: `{selected_expiry}`")
st.dataframe(merged_df, use_container_width=True)

# 🔍 Breakout Zones
df_ce_top = df_ce.sort_values(by="Chg_OI_CE", ascending=False).head(3)
df_pe_top = df_pe.sort_values(by="Chg_OI_PE", ascending=False).head(3)

st.markdown("### 🔍 Breakout Zones")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🚀 Top CALL Breakout")
    for _, row in df_ce_top.iterrows():
        st.success(f"Strike: {row['strikePrice']} | OI↑: {int(row['Chg_OI_CE'])} | Vol: {int(row['Vol_CE'])}")
with col2:
    st.subheader("🔻 Top PUT Breakout")
    for _, row in df_pe_top.iterrows():
        st.success(f"Strike: {row['strikePrice']} | OI↑: {int(row['Chg_OI_PE'])} | Vol: {int(row['Vol_PE'])}")

# 🛑📈 Support & Resistance
df_ce_score = df_ce.copy()
df_pe_score = df_pe.copy()
df_ce_score["score"] = df_ce_score["Chg_OI_CE"] + df_ce_score["Vol_CE"]
df_pe_score["score"] = df_pe_score["Chg_OI_PE"] + df_pe_score["Vol_PE"]

resistance_strike = df_ce_score.sort_values("score", ascending=False).iloc[0]["strikePrice"]
support_strike = df_pe_score.sort_values("score", ascending=False).iloc[0]["strikePrice"]

st.markdown("### 🛑📈 Support & Resistance Zone")
col1, col2 = st.columns(2)
with col1:
    st.error(f"📉 Strong Support at **{int(support_strike)}** (PUT OI + Vol)")
with col2:
    st.success(f"📈 Strong Resistance at **{int(resistance_strike)}** (CALL OI + Vol)")

# 🤖 Auto Trade Suggestion
st.markdown("### 🤖 Auto Trade Suggestion")

if support_strike == resistance_strike:
    action = "⚠️ Avoid Trade"
    reason = f"Support & Resistance both at {support_strike} → Range-bound zone"
elif resistance_strike > support_strike and pcr < 0.8:
    action = "🟥 BUY PUT"
    reason = f"PCR = {pcr} (Bearish) & Resistance > Support → Downside expected"
elif support_strike > resistance_strike and pcr > 1.2:
    action = "🟩 BUY CALL"
    reason = f"PCR = {pcr} (Bullish) & Support > Resistance → Upside expected"
else:
    action = "🔄 Wait or Sell Options"
    reason = f"Indecisive structure – PCR = {pcr} | Support ≠ Resistance"

st.info(f"**Suggested Action:** {action}")
st.caption(f"📌 Reason: {reason}")
