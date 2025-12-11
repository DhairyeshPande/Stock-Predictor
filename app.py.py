import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objects as go


# ------------------ FETCH DATA ------------------
def fetch_stock_data(symbol, start="2005-01-01", end="2025-01-01"):
    df = yf.download(symbol, start=start, end=end)

    df["MA20"] = ta.trend.sma_indicator(df["Close"], 20)
    df["MA50"] = ta.trend.sma_indicator(df["Close"], 50)
    df["RSI"] = ta.momentum.rsi(df["Close"], 14)
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["ATR_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["ADX_14"] = ta.trend.adx(df["High"], df["Low"], df["Close"], 14)
    df["BB_WIDTH"] = ta.volatility.bollinger_wband(df["Close"], 20, 2)

    df.dropna(inplace=True)
    return df


# ------------------ CREATE SEQUENCES ------------------
def create_sequences(data, window=100):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i-window:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# ------------------ LOAD MODEL + SCALER ------------------
def load_model_and_scaler(symbol):
    model = load_model(f"models/{symbol}_model.h5")
    scaler = pickle.load(open(f"models/{symbol}_scaler.pkl", "rb"))
    return model, scaler


# ------------------ PREDICT TOMORROW ------------------
def predict_tomorrow(symbol):
    df = fetch_stock_data(symbol)
    features = ["Close","MA20","MA50","RSI","Volume_MA20","ATR_14","ADX_14","BB_WIDTH"]
    df = df[features]

    model, scaler = load_model_and_scaler(symbol)

    scaled = scaler.transform(df)
    last_seq = scaled[-100:].reshape(1, 100, len(features))

    pred_scaled = model.predict(last_seq, verbose=0)[0][0]

    # invert scaling
    close_scale = scaler.scale_[0]
    close_min = scaler.min_[0]

    pred_real = pred_scaled / close_scale + close_min
    return pred_real


# ------------------ PREDICT RANGE ------------------
def predict_range(symbol, start, end):
    df = fetch_stock_data(symbol, start, end)
    features = ["Close","MA20","MA50","RSI","Volume_MA20","ATR_14","ADX_14","BB_WIDTH"]

    df_feat = df[features]
    model, scaler = load_model_and_scaler(symbol)

    scaled = scaler.transform(df_feat)
    x, y_true = create_sequences(scaled)

    preds_scaled = model.predict(x, verbose=0).flatten()

    close_scale = scaler.scale_[0]
    close_min = scaler.min_[0]

    preds = preds_scaled / close_scale + close_min
    y_true = np.array(y_true) / close_scale + close_min

    return preds, y_true


# ------------------ STREAMLIT UI ------------------
st.title("📈 Stock Predictor (Keras/TensorFlow)")

tab1, tab2 = st.tabs(["Predict Tomorrow", "Predict Range"])


# ---------- TAB 1: Tomorrow Prediction ----------
with tab1:
    symbol = st.text_input("Enter NSE Symbol", "HDFCBANK.NS")

    if st.button("Predict Tomorrow"):
        try:
            pred = predict_tomorrow(symbol)
            st.success(f"Predicted price for tomorrow: ₹{pred:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")


# ---------- TAB 2: Range Prediction ----------
with tab2:
    symbol_r = st.text_input("Symbol", "HDFCBANK.NS", key="r")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    if st.button("Predict Range"):
        try:
            preds, actual = predict_range(symbol_r, str(start), str(end))

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actual, name="Actual Price"))
            fig.add_trace(go.Scatter(y=preds, name="Predicted Price"))

            fig.update_layout(
                title=f"Actual vs Predicted — {symbol_r}",
                xaxis_title="Days",
                yaxis_title="Price (INR)"
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error: {e}")
