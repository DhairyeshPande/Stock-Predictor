import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import ta
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from keras.models import load_model
from datetime import datetime
import time

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# ==========================================================
#   FETCH STOCK DATA  (Yahoo → FDR fallback)
# ==========================================================
def fetch_stock_data(symbol, start, end, retries=3):
    df = None

    # ---- Try Yahoo Finance first ----
    for attempt in range(retries):
        try:
            df = yf.download(
                symbol, start=start, end=end,
                interval="1d", progress=False, threads=False
            )
            if df is not None and not df.empty:
                break
        except:
            pass
        time.sleep(1)

    # ---- If Yahoo fails, use FDR ----
    if df is None or df.empty:
        try:
            s = symbol.replace(".NS", "")
            df = fdr.DataReader(s, start, end)
            if df is None or df.empty:
                return None

            df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)
        except:
            return None

    # ---- Add technical indicators ----
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        df["MA20"] = ta.trend.sma_indicator(close, window=20)
        df["MA50"] = ta.trend.sma_indicator(close, window=50)
        df["RSI"] = ta.momentum.rsi(close, window=14)
        df["Volume_MA20"] = volume.rolling(20).mean()
        df["ATR_14"] = ta.volatility.average_true_range(high, low, close, 14)
        df["ADX_14"] = ta.trend.adx(high, low, close, 14)
        df["BB_WIDTH"] = ta.volatility.bollinger_wband(close, 20, 2)

        df.dropna(inplace=True)
    except:
        return None

    return df


# ==========================================================
#   HELPERS — SEQUENCE CREATION
# ==========================================================
def create_sequences(data, window=100):
    X, Y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)


# ==========================================================
#   LOAD MODELS FROM /models
# ==========================================================
def load_models(symbols, model_dir="models"):
    out = []
    for s in symbols:
        model_path = f"{model_dir}/{s}_model.h5"
        scaler_path = f"{model_dir}/{s}_scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Missing model files for: {s}")
            continue

        model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        out.append({"symbol": s, "model": model, "scaler": scaler})
    return out


# ==========================================================
#   PREDICT TOMORROW PRICE
# ==========================================================
def predict_tomorrow(symbol, models_group, window=100):
    df = fetch_stock_data(symbol, "2010-01-01", "2025-01-01")
    if df is None or df.empty:
        return None

    features = ["Close", "MA20", "MA50", "RSI",
                "Volume_MA20", "ATR_14", "ADX_14", "BB_WIDTH"]

    df = df[features].dropna()
    if len(df) < window:
        return None

    best_pred = None
    best_r2 = -1

    for entry in models_group:
        model = entry["model"]
        scaler = entry["scaler"]

        arr = scaler.transform(df)
        last_window = arr[-window:].reshape(1, window, arr.shape[1])

        pred_scaled = model.predict(last_window, verbose=0).flatten()[0]
        pred = pred_scaled / scaler.scale_[0] + scaler.min_[0]

        X_recent, y_true = create_sequences(arr, window)
        y_pred_recent = model.predict(X_recent, verbose=0).flatten()

        y_pred_recent = y_pred_recent / scaler.scale_[0] + scaler.min_[0]
        y_true = y_true / scaler.scale_[0] + scaler.min_[0]

        r2 = r2_score(y_true, y_pred_recent)

        if r2 > best_r2:
            best_r2 = r2
            best_pred = pred

    return best_pred


# ==========================================================
#   PREDICT RANGE
# ==========================================================
def predict_range(symbol, start, end, models_group, window=100):
    df = fetch_stock_data(symbol, start, end)
    if df is None or df.empty:
        return None, None, None

    features = ["Close", "MA20", "MA50", "RSI",
                "Volume_MA20", "ATR_14", "ADX_14", "BB_WIDTH"]

    df = df[features].dropna()
    if len(df) < window:
        return None, None, None

    best_result = None
    best_r2 = -1

    for entry in models_group:
        model = entry["model"]
        scaler = entry["scaler"]

        try:
            arr = scaler.transform(df)
        except:
            continue

        X_new, y_true = create_sequences(arr, window)
        if len(X_new) == 0:
            continue

        pred_scaled = model.predict(X_new, verbose=0).flatten()
        pred = pred_scaled / scaler.scale_[0] + scaler.min_[0]
        y_true = y_true / scaler.scale_[0] + scaler.min_[0]

        r2 = r2_score(y_true, pred)

        if r2 > best_r2:
            best_r2 = r2
            best_result = (pred, y_true, df)

    return best_result


# ==========================================================
#   MODEL GROUPS
# ==========================================================
group_less = ["MAXHEALTH.NS", "IDEA.NS", "ITC.NS", "TATAMOTORS.NS"]
group_mid = ["ICICIBANK.NS", "HDFCBANK.NS", "SIEMENS.NS", "HEROMOTOCO.NS"]
group_more = ["MRF.NS", "BAJFINANCE.NS", "POWERINDIA.NS", "PAGEIND.NS"]

all_models = {
    "low": load_models(group_less),
    "mid": load_models(group_mid),
    "high": load_models(group_more),
}


# ==========================================================
#   UI — STREAMLIT
# ==========================================================
st.title("📈 Stock Price Predictor")

tab1, tab2 = st.tabs(["🔮 Predict Tomorrow", "📆 Predict Price Range"])

# ------------------------
# TAB 1 – Tomorrow
# ------------------------
with tab1:
    st.subheader("Predict Tomorrow's Closing Price")

    symbol = st.text_input("Enter NSE Symbol (e.g., HDFCBANK.NS)", "")

    if st.button("Predict Tomorrow"):
        if symbol.strip() == "":
            st.error("Please enter a valid symbol.")
        else:
            price = predict_tomorrow(symbol, all_models["mid"])
            if price is None:
                st.error("Prediction failed.")
            else:
                st.success(f"Predicted closing price: **₹{price:.2f}**")


# ------------------------
# TAB 2 – Range Prediction
# ------------------------
with tab2:
    st.subheader("Predict Price Range")

    symbol = st.text_input("Symbol", key="range_symbol")
    start = st.text_input("Start Date (YYYY-MM-DD)", "2015-01-01")
    end = st.text_input("End Date (YYYY-MM-DD)", "2025-01-01")

    if st.button("Predict Range"):
        result = predict_range(symbol, start, end, all_models["mid"])
        if result is None:
            st.error("Prediction failed.")
        else:
            pred, truth, dfp = result
            st.success("Prediction Completed")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=truth, name="Actual"))
            fig.add_trace(go.Scatter(y=pred, name="Predicted"))

            fig.update_layout(
                title=f"{symbol} – Price Prediction",
                height=500,
                xaxis_title="Days",
                yaxis_title="Price (₹)"
            )

            st.plotly_chart(fig, use_container_width=True)
