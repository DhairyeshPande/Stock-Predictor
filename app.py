import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --------------------------------------------------------------------
# SAFE YFINANCE FETCH FUNCTION
# --------------------------------------------------------------------

def safe_fetch(symbol, start="2010-01-01", end=None, retries=4, delay=1.2):
    """
    yfinance-only safe downloader with retries.
    Never crashes the app.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    s = symbol.upper().strip()
    if not s.endswith(".NS"):
        s = s + ".NS"

    for attempt in range(retries):
        try:
            df = yf.download(
                s,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.dropna(inplace=True)
                return df

        except Exception:
            pass

        time.sleep(delay)
        delay *= 1.5

    return None

# --------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------

def add_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["MA20"] = ta.trend.sma_indicator(close, window=20)
    df["MA50"] = ta.trend.sma_indicator(close, window=50)
    df["RSI"] = ta.momentum.rsi(close, window=14)
    df["Volume_MA20"] = volume.rolling(20).mean()
    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["ADX_14"] = ta.trend.adx(high, low, close, window=14)
    df["BB_WIDTH"] = ta.volatility.bollinger_wband(close, window=20, window_dev=2)

    df.dropna(inplace=True)
    return df

# --------------------------------------------------------------------
# PREPARE SEQUENCES
# --------------------------------------------------------------------

def create_sequences(data, window=100):
    X, Y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)


# --------------------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------------------

def load_models(symbols, model_dir="models"):
    loaded = []
    for s in symbols:
        try:
            model_path = os.path.join(model_dir, f"{s}_model.h5")
            scaler_path = os.path.join(model_dir, f"{s}_scaler.pkl")

            model = load_model(model_path)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            loaded.append({"symbol": s, "model": model, "scaler": scaler})

        except Exception as e:
            print("Failed loading:", s, e)

    return loaded

# --------------------------------------------------------------------
# SELECT MODEL BASED ON PRICE GROUP
# --------------------------------------------------------------------

def get_price_group(price):
    price = float(price)
    if price < 800:
        return "less"
    elif price < 3000:
        return "mid"
    return "more"


# --------------------------------------------------------------------
# PREDICT TOMORROW
# --------------------------------------------------------------------

def predict_tomorrow(symbol, models, window=100):
    df = safe_fetch(symbol, start="2015-01-01")
    if df is None or df.empty:
        return None

    df = add_indicators(df)
    if len(df) < window:
        return None

    features = ["Close", "MA20", "MA50", "RSI", "Volume_MA20",
                "ATR_14", "ADX_14", "BB_WIDTH"]
    df = df[features].dropna()
    if len(df) < window:
        return None

    best_r2 = -999
    best_pred = None

    for item in models:
        model = item["model"]
        scaler = item["scaler"]

        try:
            data_scaled = scaler.transform(df)
        except:
            continue

        last_window = data_scaled[-window:].reshape(1, window, df.shape[1])
        y_scaled = model.predict(last_window, verbose=0).flatten()[0]

        close_scale = scaler.scale_[0]
        close_min = scaler.min_[0]
        y_pred = y_scaled / close_scale + close_min

        best_pred = y_pred

    return best_pred


# --------------------------------------------------------------------
# RANGE PREDICTION
# --------------------------------------------------------------------

def predict_range(symbol, start, end, models, window=100):
    df = safe_fetch(symbol, start=start, end=end)
    if df is None or df.empty:
        return None, None, None

    df = add_indicators(df)

    features = ["Close", "MA20", "MA50", "RSI", "Volume_MA20",
                "ATR_14", "ADX_14", "BB_WIDTH"]
    df = df[features].dropna()

    if len(df) < window:
        return None, None, None

    best_pred = None
    best_true = None

    for item in models:
        model = item["model"]
        scaler = item["scaler"]

        try:
            scaled = scaler.transform(df)
        except:
            continue

        X, Y = create_sequences(scaled, window)
        if len(X) == 0:
            continue

        yp_scaled = model.predict(X, verbose=0).flatten()

        close_scale = scaler.scale_[0]
        close_min = scaler.min_[0]
        yp = yp_scaled / close_scale + close_min
        yt = Y / close_scale + close_min

        best_pred = yp
        best_true = yt

    return best_pred, best_true, df


# --------------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------------

st.set_page_config(page_title="NSE Stock Predictor", layout="wide")
st.title("📊 Stock Price Predictor")

tab1, tab2 = st.tabs(["📆 Predict Range", "📅 Predict Tomorrow"])

group_less = ["MAXHEALTH.NS", "IDEA.NS", "ITC.NS", "TATAMOTORS.NS"]
group_mid = ["ICICIBANK.NS", "HDFCBANK.NS", "SIEMENS.NS", "HEROMOTOCO.NS"]
group_more = ["MRF.NS", "BAJFINANCE.NS", "POWERINDIA.NS", "PAGEIND.NS"]

all_models = {
    "less": load_models(group_less),
    "mid": load_models(group_mid),
    "more": load_models(group_more),
}


# --------------------------------------------------------------------
# TAB 2 — PREDICT TOMORROW
# --------------------------------------------------------------------

with tab2:
    st.header("Predict Tomorrow's Closing Price")

    symbol = st.text_input("Enter NSE Symbol", placeholder="HDFCBANK")
    if symbol:
        full_symbol = symbol.upper().strip() + ".NS"

        if st.button("Predict Tomorrow"):
            df_check = safe_fetch(full_symbol, start="2023-01-01")
            if df_check is None or df_check.empty:
                st.error("Failed to fetch data. Try again.")
            else:
                latest = float(df_check["Close"].iloc[-1])
                group = get_price_group(latest)

                pred = predict_tomorrow(full_symbol, all_models[group])
                if pred:
                    st.success(f"Predicted close price for **{symbol}**: ₹{pred:.2f}")
                else:
                    st.error("Prediction failed.")


# --------------------------------------------------------------------
# TAB 1 — RANGE PREDICTION
# --------------------------------------------------------------------

with tab1:
    st.header("Predict Price Range")

    symbol2 = st.text_input("Enter NSE Symbol (for range)", placeholder="HDFCBANK")
    start = st.date_input("Start Date", value=datetime(2020, 1, 1))
    end = st.date_input("End Date", value=datetime(2025, 1, 1))

    if st.button("Predict & Show Graph"):
        full_symbol = symbol2.upper().strip() + ".NS"

        df_check = safe_fetch(full_symbol, start="2023-01-01")
        if df_check is None or df_check.empty:
            st.error("Failed to fetch recent price data. Try again.")
        else:
            latest = float(df_check["Close"].iloc[-1])
            group = get_price_group(latest)

            y_pred, y_true, df = predict_range(full_symbol, str(start), str(end), all_models[group])

            if y_pred is None:
                st.error("Prediction failed.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_true, name="Actual"))
                fig.add_trace(go.Scatter(y=y_pred, name="Predicted"))

                fig.update_layout(title=f"{symbol2} Price Range Prediction",
                                  xaxis_title="Days",
                                  yaxis_title="Price (₹)",
                                  height=500)

                st.plotly_chart(fig, use_container_width=True)
