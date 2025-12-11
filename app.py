import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import pickle
import os
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime


st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("(Keras/TensorFlow)")


# ----------------------------------------------------------
#  FETCH STOCK DATA
# ----------------------------------------------------------
def fetch_stock_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, threads=False)

        if df is None or df.empty:
            return None

        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]
        low = df["Low"]

        df["MA20"] = ta.trend.sma_indicator(close, window=20)
        df["MA50"] = ta.trend.sma_indicator(close, window=50)
        df["RSI"] = ta.momentum.rsi(close, window=14)
        df["Volume_MA20"] = volume.rolling(window=20).mean()
        df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["ADX_14"] = ta.trend.adx(high, low, close, window=14)
        df["BB_WIDTH"] = ta.volatility.bollinger_wband(close, window=20, window_dev=2)

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print("YFinance Error:", e)
        return None


# ----------------------------------------------------------
#  CREATE SEQUENCES
# ----------------------------------------------------------
def create_sequences(data, window=100):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i - window:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# ----------------------------------------------------------
#  LOAD MODELS
# ----------------------------------------------------------
def load_models(symbols):
    models = []
    for s in symbols:
        model_path = f"models/{s}_model.h5"
        scaler_path = f"models/{s}_scaler.pkl"

        if not os.path.exists(model_path):
            continue

        model = load_model(model_path)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        models.append({"symbol": s, "model": model, "scaler": scaler})

    return models


# ----------------------------------------------------------
#  PREDICT TOMORROW PRICE
# ----------------------------------------------------------
def predict_tomorrow(symbol, trained_models, window=100):
    df = fetch_stock_data(symbol, "2010-01-01", datetime.today().strftime("%Y-%m-%d"))

    if df is None or df.empty:
        st.error("Failed to fetch stock data.")
        return None

    features = ["Close", "MA20", "MA50", "RSI", "Volume_MA20", "ATR_14", "ADX_14", "BB_WIDTH"]
    df = df[features].dropna()

    best_r2 = -np.inf
    best_pred = None

    for entry in trained_models:
        model = entry["model"]
        scaler = entry["scaler"]

        try:
            data_scaled = scaler.transform(df)
        except:
            continue

        if len(data_scaled) < window:
            continue

        last_window = data_scaled[-window:].reshape(1, window, data_scaled.shape[1])
        y_scaled = model.predict(last_window, verbose=0).flatten()[0]

        # inverse transform
        y_pred = y_scaled / scaler.scale_[0] + scaler.min_[0]

        # compute recent R2
        x_recent, y_true = create_sequences(data_scaled, window)
        pred_recent = model.predict(x_recent, verbose=0).flatten()

        pred_recent = pred_recent / scaler.scale_[0] + scaler.min_[0]
        y_true = np.array(y_true) / scaler.scale_[0] + scaler.min_[0]

        r2 = r2_score(y_true, pred_recent)

        if r2 > best_r2:
            best_r2 = r2
            best_pred = y_pred

    return best_pred


# ----------------------------------------------------------
#  PREDICT RANGE (with graph)
# ----------------------------------------------------------
def predict_range(symbol, start, end, trained_models, window=100):
    df = fetch_stock_data(symbol, start, end)

    if df is None or df.empty:
        st.error("Failed to fetch stock data.")
        return None, None

    features = ["Close", "MA20", "MA50", "RSI", "Volume_MA20", "ATR_14", "ADX_14", "BB_WIDTH"]
    df = df[features].dropna()

    best_r2 = -np.inf
    best_y_pred = None
    best_y_true = None

    for entry in trained_models:
        model = entry["model"]
        scaler = entry["scaler"]

        try:
            scaled = scaler.transform(df)
        except:
            continue

        x_new, y_true = create_sequences(scaled, window)
        if len(x_new) == 0:
            continue

        y_scaled = model.predict(x_new, verbose=0).flatten()

        # inverse scale
        y_pred = y_scaled / scaler.scale_[0] + scaler.min_[0]
        y_true_rescaled = y_true / scaler.scale_[0] + scaler.min_[0]

        r2 = r2_score(y_true_rescaled, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_y_pred = y_pred
            best_y_true = y_true_rescaled

    return best_y_pred, best_y_true


# ----------------------------------------------------------
# MODEL GROUPS
# ----------------------------------------------------------
group_less = ["MAXHEALTH.NS", "IDEA.NS", "ITC.NS", "TATAMOTORS.NS"]
group_mid = ["ICICIBANK.NS", "HDFCBANK.NS", "SIEMENS.NS", "HEROMOTOCO.NS"]
group_more = ["MRF.NS", "BAJFINANCE.NS", "POWERINDIA.NS", "PAGEIND.NS"]

all_models = {
    "less": load_models(group_less),
    "mid": load_models(group_mid),
    "more": load_models(group_more),
}


# ----------------------------------------------------------
#   STREAMLIT UI
# ----------------------------------------------------------
tab1, tab2 = st.tabs(["Predict Tomorrow", "Predict Range"])


# ==========================================================
#  TAB 1 — TOMORROW PRICE
# ==========================================================
with tab1:
    st.header("Predict Tomorrow")

    symbol = st.text_input("Symbol", "HDFCBANK.NS")

    if st.button("Predict Tomorrow"):
        latest = yf.download(symbol, period="5d")
        if latest.empty:
            st.error("Invalid symbol or no recent data.")
        else:
            price = float(latest["Close"].iloc[-1])

            if price <= 800:
                models_group = all_models["less"]
            elif price <= 3000:
                models_group = all_models["mid"]
            else:
                models_group = all_models["more"]

            pred = predict_tomorrow(symbol, models_group)

            if pred is not None:
                st.success(f"Tomorrow's predicted close price: ₹{pred:.2f}")
            else:
                st.error("Prediction failed.")


# ==========================================================
#  TAB 2 — RANGE PREDICTION
# ==========================================================
with tab2:
    st.header("Predict Range")

    symbol = st.text_input("Symbol", "HDFCBANK.NS", key="range_symbol")
    start = st.text_input("Start Date", "2015-01-01")
    end = st.text_input("End Date", "2025-01-01")

    if st.button("Predict Range"):
        latest = yf.download(symbol, period="5d")

        if latest.empty:
            st.error("Invalid symbol.")
        else:
            price = float(latest["Close"].iloc[-1])

            if price <= 800:
                models_group = all_models["less"]
            elif price <= 3000:
                models_group = all_models["mid"]
            else:
                models_group = all_models["more"]

            y_pred, y_true = predict_range(symbol, start, end, models_group)

            if y_pred is None:
                st.error("Prediction failed.")
            else:
                # reshape to 1D
                y_pred = np.array(y_pred).reshape(-1)
                y_true = np.array(y_true).reshape(-1)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_true, name="Actual"))
                fig.add_trace(go.Scatter(y=y_pred, name="Predicted"))
                fig.update_layout(title=f"{symbol} Prediction", height=500)

                st.plotly_chart(fig, use_container_width=True)
