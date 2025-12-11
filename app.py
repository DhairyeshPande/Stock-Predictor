# app.py
import os
import time
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import ta
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from keras.models import load_model

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor")

# -----------------------------
# SAFE DATA FETCH (YF -> FDR fallback)
# -----------------------------
def fetch_stock_data(symbol, start="2005-01-01", end="2025-01-01", retries=3, backoff=1.0):
    """
    Tries Yahoo Finance first. If no data or rate-limited, falls back to FinanceDataReader (FDR).
    Returns dataframe with indicators added or None on failure.
    """
    df = None
    # try yahoo finance with retries
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, interval="1d", progress=False, threads=False)
            if df is not None and not df.empty:
                break
        except Exception as e:
            # keep trying
            print("Yahoo attempt", attempt + 1, "failed:", e)
        time.sleep(backoff)

    # fallback to FinanceDataReader if yahoo failed or returned empty
    if df is None or df.empty:
        try:
            fdr_sym = symbol.replace(".NS", "")
            df = fdr.DataReader(fdr_sym, start, end)
            if df is None or df.empty:
                return None
            # FDR already has columns named Open/High/Low/Close/Volume usually
            # ensure consistent column names
            df = df.rename(columns={c: c for c in df.columns if c in df.columns})
        except Exception as e:
            print("FDR fallback failed:", e)
            return None

    # compute indicators (make sure required columns exist)
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        df["MA20"] = ta.trend.sma_indicator(close, window=20)
        df["MA50"] = ta.trend.sma_indicator(close, window=50)
        df["RSI"] = ta.momentum.rsi(close, window=14)
        df["Volume_MA20"] = volume.rolling(window=20).mean()
        df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["ADX_14"] = ta.trend.adx(high, low, close, window=14)
        df["BB_WIDTH"] = ta.volatility.bollinger_wband(close, window=20, window_dev=2)

        df.dropna(inplace=True)
    except Exception as e:
        print("Indicator calculation failed:", e)
        return None

    # restrict to requested date range (sometimes extended by FDR)
    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    except Exception:
        pass

    if df.empty:
        return None
    return df


# -----------------------------
# SEQUENCE CREATION
# -----------------------------
def create_sequences(data, window=100):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i - window:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# -----------------------------
# LOAD MODELS (safe)
# -----------------------------
def load_models(symbols, model_dir="models"):
    loaded = []
    for s in symbols:
        model_path = os.path.join(model_dir, f"{s}_model.h5")
        scaler_path = os.path.join(model_dir, f"{s}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Missing model/scaler for {s} -> skipping")
            continue

        try:
            model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model file {model_path}: {e}")
            continue

        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading scaler {scaler_path}: {e}")
            continue

        loaded.append({"symbol": s, "model": model, "scaler": scaler})
    return loaded


# -----------------------------
# INVERSE SCALING helper
# -----------------------------
def inverse_scale_values(scaled_values, scaler, feature_index=0):
    """
    Given scaled values (1D or array) which were produced by scaler on the first feature,
    invert them using scaler.scale_[feature_index] and scaler.min_[feature_index].
    Formula: original = (scaled - min_) / scale_
    """
    scaled = np.array(scaled_values)
    min_ = scaler.min_[feature_index]
    scale_ = scaler.scale_[feature_index]
    # avoid division by zero
    if scale_ == 0:
        return np.full_like(scaled, np.nan, dtype=float)
    return (scaled - min_) / scale_


# -----------------------------
# PREDICT TOMORROW
# -----------------------------
def predict_tomorrow_price(symbol, trained_models, window=100):
    # fetch extended history for features
    end = datetime.today().strftime("%Y-%m-%d")
    start = "2010-01-01"
    df = fetch_stock_data(symbol, start=start, end=end)
    if df is None or len(df) < window:
        return None

    features = ["Close", "MA20", "MA50", "RSI", "Volume_MA20", "ATR_14", "ADX_14", "BB_WIDTH"]
    df_f = df[features].dropna()
    if len(df_f) < window:
        return None

    best_r2 = -np.inf
    best_pred = None

    for entry in trained_models:
        model = entry["model"]
        scaler = entry["scaler"]

        try:
            scaled = scaler.transform(df_f)
        except Exception as e:
            print("Scaler transform failed:", e)
            continue

        # make sure we have enough samples
        if len(scaled) < window:
            continue

        last_window = scaled[-window:].reshape(1, window, scaled.shape[1])
        try:
            y_scaled = model.predict(last_window, verbose=0).flatten()[0]
        except Exception as e:
            print("Model predict failed:", e)
            continue

        # inverse scale
        y_pred = inverse_scale_values(y_scaled, scaler, feature_index=0)

        # calculate r2 on recent historical portion to choose best model
        x_recent, y_true = create_sequences(scaled, window)
        if len(x_recent) == 0:
            continue
        try:
            y_pred_recent_scaled = model.predict(x_recent, verbose=0).flatten()
        except Exception as e:
            print("Model predict recent failed:", e)
            continue

        y_pred_recent = inverse_scale_values(y_pred_recent_scaled, scaler, 0)
        y_true_rescaled = inverse_scale_values(y_true, scaler, 0)

        # safe r2
        try:
            r2 = r2_score(y_true_rescaled, y_pred_recent)
        except Exception:
            r2 = -np.inf

        if r2 > best_r2:
            best_r2 = r2
            best_pred = float(y_pred)

    return best_pred


# -----------------------------
# PREDICT RANGE
# -----------------------------
def predict_new_stock_best(symbol, start, end, trained_models, window=100):
    df = fetch_stock_data(symbol, start=start, end=end)
    if df is None:
        return None, None, None

    features = ["Close", "MA20", "MA50", "RSI", "Volume_MA20", "ATR_14", "ADX_14", "BB_WIDTH"]
    df_f = df[features].dropna()
    if len(df_f) < window:
        return None, None, None

    best_r2 = -np.inf
    best_result = None

    for entry in trained_models:
        model = entry["model"]
        scaler = entry["scaler"]

        try:
            scaled = scaler.transform(df_f)
        except Exception:
            continue

        x_new, y_true = create_sequences(scaled, window)
        if len(x_new) == 0:
            continue

        try:
            y_pred_scaled = model.predict(x_new, verbose=0).flatten()
        except Exception:
            continue

        y_pred = inverse_scale_values(y_pred_scaled, scaler, 0)
        y_true_rescaled = inverse_scale_values(y_true, scaler, 0)

        try:
            r2 = r2_score(y_true_rescaled, y_pred)
        except Exception:
            r2 = -np.inf

        if r2 > best_r2:
            best_r2 = r2
            best_result = {"symbol": entry["symbol"], "r2": r2, "y_pred": y_pred, "y_true": y_true_rescaled, "df": df_f}

    if best_result:
        return best_result["y_pred"], best_result["y_true"], best_result["df"]
    return None, None, None


# -----------------------------
# App config: model groups and loading
# -----------------------------
group_less = ["MAXHEALTH.NS", "IDEA.NS", "ITC.NS", "TATAMOTORS.NS"]
group_mid = ["ICICIBANK.NS", "HDFCBANK.NS", "SIEMENS.NS", "HEROMOTOCO.NS"]
group_more = ["MRF.NS", "BAJFINANCE.NS", "POWERINDIA.NS", "PAGEIND.NS"]

all_models = {
    "group_less": load_models(group_less),
    "group_mid": load_models(group_mid),
    "group_more": load_models(group_more),
}

# -----------------------------
# UI
# -----------------------------
tab1, tab2 = st.tabs(["📆 Predict Range", "📅 Predict Tomorrow"])


# ---------- Predict Tomorrow ----------
with tab2:
    st.header("Predict Tomorrow's Price")

    user_symbol = st.text_input("Enter NSE Symbol (e.g. HDFCBANK)", value="HDFCBANK")
    if user_symbol.strip() == "":
        st.info("Enter a symbol like HDFCBANK (without .NS) or HDFCBANK.NS")
    symbol = user_symbol.strip().upper()
    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"

    if st.button("Predict"):
        # use safe fetch for recent price (10 days)
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_df = fetch_stock_data(symbol, start=start_date, end=end_date)
        if recent_df is None or recent_df.empty:
            st.error("Failed to fetch recent price data. Try again later.")
        else:
            latest_price = float(recent_df["Close"].iloc[-1])
            # choose model group
            if latest_price < 800:
                models = all_models["group_less"]
            elif latest_price < 3000:
                models = all_models["group_mid"]
            else:
                models = all_models["group_more"]

            if not models:
                st.error("No pre-trained models available for the chosen group. Upload models into /models.")
            else:
                pred = predict_tomorrow_price(symbol, models)
                if pred is None:
                    st.error("Prediction failed. Possibly insufficient data or model mismatch.")
                else:
                    st.success(f"📌 Tomorrow's predicted close price for {symbol.replace('.NS','')}: ₹{pred:.2f}")


# ---------- Predict Range ----------
with tab1:
    st.header("Predict Price Range")

    user_symbol_range = st.text_input("Enter NSE Symbol (for range)", value="HDFCBANK", key="range_sym")
    if user_symbol_range.strip() == "":
        st.info("Enter a symbol like HDFCBANK (without .NS) or HDFCBANK.NS")
    symbol_range = user_symbol_range.strip().upper()
    if not symbol_range.endswith(".NS"):
        symbol_range = symbol_range + ".NS"

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    with col2:
        end = st.date_input("End Date", value=pd.to_datetime(datetime.today().strftime("%Y-%m-%d")))

    if st.button("Predict & Show Graph"):
        # fetch a short recent window to determine model group (avoid calling yf directly)
        recent_start = (pd.to_datetime(end) - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_end = pd.to_datetime(end).strftime("%Y-%m-%d")
        recent_df = fetch_stock_data(symbol_range, start=recent_start, end=recent_end)
        if recent_df is None or recent_df.empty:
            st.error("Failed to fetch recent price data. Try again later.")
        else:
            latest_price = float(recent_df["Close"].iloc[-1])
            if latest_price < 800:
                models = all_models["group_less"]
            elif latest_price < 3000:
                models = all_models["group_mid"]
            else:
                models = all_models["group_more"]

            if not models:
                st.error("No pre-trained models available for the chosen group. Upload models into /models.")
            else:
                y_pred, y_true, df_used = predict_new_stock_best(symbol_range, str(start), str(end), models)
                if y_pred is None:
                    st.error("Prediction failed. Check data availability or model compatibility.")
                else:
                    # ensure 1d arrays for plotting
                    y_pred = np.array(y_pred).reshape(-1)
                    y_true = np.array(y_true).reshape(-1)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_true, name="Actual", line=dict(width=2)))
                    fig.add_trace(go.Scatter(y=y_pred, name="Predicted", line=dict(width=2)))

                    fig.update_layout(
                        title=f"{symbol_range.replace('.NS','')} - Actual vs Predicted",
                        height=500,
                        xaxis_title="Trading Days",
                        yaxis_title="Price (INR)",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )

                    st.plotly_chart(fig, use_container_width=True)
