import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import datetime as dt
import requests
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math
import json
import os

# ===========================
# CONFIG
# ===========================
CONFIG = {
    "MODE": "proactive",               # safe / proactive / aggressive
    "SYMBOL": "EURUSD",
    "PRIMARY_TF": mt5.TIMEFRAME_M5,
    "HTF": mt5.TIMEFRAME_M15,
    "BASE_FAST_SMA": 10,
    "BASE_SLOW_SMA": 50,
    "BASE_RSI": 14,
    "ATR_PERIOD": 14,
    "RISK_PER_TRADE": 0.006,
    "SL_ATR_MULT": 1.5,
    "TP_ATR_MULT": 2.5,
    "MODEL_RF_FILE": "rf_model.pkl",
    "MODEL_DEEP_FILE": "deep_model.h5",
    "SEQUENCE_LEN": 32,
    "MODEL_PROB_THRESHOLD": 0.30,
    "DRY_RUN": False,
    "LOOP_DELAY": 20,
    "MAGIC": 424242,
    "COOLDOWN_SECS": 300,
    "TELEGRAM_ENABLED": False,
    "TELEGRAM_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "SL_DOLLAR": 15,
    "USE_DOLLAR_SL": True,
    "TP_DOLLAR": 10,
    "USE_DOLLAR_TP": True,
    "RETRAIN_INTERVAL_HOURS": 12,
    "MIN_SAMPLES_RF": 50,
    "MIN_SAMPLES_DEEP": 200,
    "TRAINING_SAMPLES_FILE": "training_samples.csv",
    "TRADE_LOG_FILE": "trade_log.csv",
    "RETRAIN_ON_START": False,
}


def apply_mode_settings():
    m = CONFIG["MODE"].lower()
    if m == "safe":
        CONFIG.update({"MODEL_PROB_THRESHOLD": 0.45, "RISK_PER_TRADE": 0.005, "SL_ATR_MULT": 1.6, "TP_ATR_MULT": 2.8})
    elif m == "proactive":
        CONFIG.update({"MODEL_PROB_THRESHOLD": 0.25, "RISK_PER_TRADE": 0.01, "SL_ATR_MULT": 1.3, "TP_ATR_MULT": 2.1})
    elif m == "aggressive":
        CONFIG.update({"MODEL_PROB_THRESHOLD": 0.15, "RISK_PER_TRADE": 0.015, "SL_ATR_MULT": 1.0, "TP_ATR_MULT": 1.8})
    else:
        CONFIG["MODE"] = "safe"
        apply_mode_settings()

apply_mode_settings()


# ===========================
# UTILITIES
# ===========================
def send_telegram(msg: str):
    if CONFIG.get("TELEGRAM_ENABLED", False):
        try:
            url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
            requests.post(url, data={"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": msg}, timeout=5)
        except Exception as e:
            print("Telegram error:", e)


def get_data(symbol, timeframe, bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def calc_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-12)
    return 100 - (100 / (1 + rs))


# ===========================
# MODEL LOADING
# ===========================
def load_models():
    rf, deep = None, None
    if os.path.exists(CONFIG["MODEL_RF_FILE"]):
        try:
            rf = joblib.load(CONFIG["MODEL_RF_FILE"])
            print("✅ RF model loaded.")
        except Exception as e:
            print("⚠️ RF load error:", e)
    else:
        print("⚠️ RF model file not found.")

    if os.path.exists(CONFIG["MODEL_DEEP_FILE"]):
        try:
            deep = load_model(CONFIG["MODEL_DEEP_FILE"], compile=False)
            print("✅ Deep model loaded.")
        except Exception as e:
            print("⚠️ Deep load error:", e)
    else:
        print("⚠️ Deep model file not found.")

    return rf, deep


def build_default_deep(input_shape, lr=1e-3):
    seq_len, n_feat = input_shape
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(seq_len, n_feat)),
        Dropout(0.2),
        Conv1D(16, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ===========================
# FEATURE ENGINEERING
# ===========================
def prepare_full_features(df_raw):
    df = df_raw.copy()
    df["ret1"] = df["close"].pct_change().fillna(0)
    df["vol"] = df.get("tick_volume", 0)
    df["sma_fast"] = df["close"].rolling(CONFIG["BASE_FAST_SMA"]).mean()
    df["sma_slow"] = df["close"].rolling(CONFIG["BASE_SLOW_SMA"]).mean()
    df["sma_diff"] = df["sma_fast"] - df["sma_slow"]
    df["sma_slope_fast"] = df["sma_fast"].diff()
    df["sma_slope_slow"] = df["sma_slow"].diff()
    df["rsi"] = compute_rsi(df["close"], CONFIG["BASE_RSI"])
    df["atr"] = calc_atr(df, CONFIG["ATR_PERIOD"])

    # simple ADX approximation
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(CONFIG["ATR_PERIOD"]).mean()
    plus_di = 100 * (plus_dm.rolling(CONFIG["ATR_PERIOD"]).sum() / (atr_series + 1e-9))
    minus_di = 100 * (minus_dm.rolling(CONFIG["ATR_PERIOD"]).sum() / (atr_series + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df["adx"] = dx.rolling(CONFIG["ATR_PERIOD"]).mean()
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


# ===========================
# PREDICTION & SIGNAL
# ===========================
def get_rf_expected_features(rf_model=None):
    return ["ret1", "sma_diff", "sma_slope_fast", "sma_slope_slow", "rsi", "atr", "adx", "vol"]


def align_rf_row(df, rf_model):
    expected = get_rf_expected_features(rf_model)
    for c in expected:
        if c not in df.columns:
            df[c] = 0.0
    return df[expected].iloc[[-1]]


def prepare_deep_seq(df, deep_model):
    seq_len = CONFIG["SEQUENCE_LEN"]
    feat_cols = get_rf_expected_features()
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    seq_df = df[feat_cols].tail(seq_len)
    if len(seq_df) < seq_len:
        pad = pd.DataFrame(0, index=range(seq_len - len(seq_df)), columns=seq_df.columns)
        seq_df = pd.concat([pad, seq_df])
    return seq_df.values


def ensemble_prediction_safe(rf_model, deep_model, df_features):
    rf_prob, deep_prob = 0.3, 0.3
    try:
        if rf_model:
            X_rf = align_rf_row(df_features, rf_model)
            rf_prob = float(rf_model.predict_proba(X_rf)[0][1])
        if deep_model:
            seq = prepare_deep_seq(df_features, deep_model)
            deep_prob = float(deep_model.predict(np.expand_dims(seq, 0), verbose=0)[0][0])
    except Exception as e:
        print("Prediction error:", e)
    return 0.4 * rf_prob + 0.6 * deep_prob


def get_signal(rf_model, deep_model, df_raw):
    df = prepare_full_features(df_raw)
    if df.empty:
        return None
    last = df.iloc[-1]
    direction = None
    if last["sma_fast"] > last["sma_slow"] and last["rsi"] > 55:
        direction = "buy"
    elif last["sma_fast"] < last["sma_slow"] and last["rsi"] < 45:
        direction = "sell"
    if not direction:
        return None
    prob = ensemble_prediction_safe(rf_model, deep_model, df)
    print(f"Signal={direction} | Prob={prob:.3f}")
    return direction if prob >= CONFIG["MODEL_PROB_THRESHOLD"] else None


# ===========================
# POSITION HELPERS
# ===========================
def has_open_position(symbol):
    try:
        pos = mt5.positions_get(symbol=symbol)
        return bool(pos)
    except Exception:
        return False


# ===========================
# PLACE TRADE
# ===========================
def place_trade(signal, df_raw):
    symbol = CONFIG["SYMBOL"]
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick info.")
        return
    entry = tick.ask if signal == "buy" else tick.bid
    atr = calc_atr(df_raw, CONFIG["ATR_PERIOD"]).iloc[-1]

    # compute SL & TP (dollar or ATR)
    if CONFIG["USE_DOLLAR_TP"]:
        pip_value = 10
        lot = 0.1
        pip_dist = CONFIG["TP_DOLLAR"] / (pip_value * lot)
        tp = entry + pip_dist * 0.0001 if signal == "buy" else entry - pip_dist * 0.0001
    else:
        tp = entry + CONFIG["TP_ATR_MULT"] * atr if signal == "buy" else entry - CONFIG["TP_ATR_MULT"] * atr

    if CONFIG["USE_DOLLAR_SL"]:
        pip_value = 10
        lot = 0.1
        pip_dist = CONFIG["SL_DOLLAR"] / (pip_value * lot)
        sl = entry - pip_dist * 0.0001 if signal == "buy" else entry + pip_dist * 0.0001
    else:
        sl = entry - CONFIG["SL_ATR_MULT"] * atr if signal == "buy" else entry + CONFIG["SL_ATR_MULT"] * atr

    if CONFIG["DRY_RUN"]:
        print(f"[DRY-RUN] {signal.upper()} {symbol} @ {entry:.5f} SL={sl:.5f} TP={tp:.5f}")
        return

    order_type = mt5.ORDER_TYPE_BUY if signal == "buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1,
        "type": order_type,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": CONFIG["MAGIC"],
        "comment": f"AI-{signal}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    print("Trade result:", result)


# ===========================
# MAIN LOOP
# ===========================
def trade_loop():
    if not mt5.initialize():
        print("MT5 initialization failed.")
        return

    rf_model, deep_model = load_models()
    print("✅ Models loaded. Starting loop...")

    last_trade_time = 0

    while True:
        try:
            df = get_data(CONFIG["SYMBOL"], CONFIG["PRIMARY_TF"])
            if df is None or df.empty:
                print("No data. Retrying...")
                time.sleep(CONFIG["LOOP_DELAY"])
                continue

            if time.time() - last_trade_time < CONFIG["COOLDOWN_SECS"]:
                print("Cooldown active.")
                time.sleep(CONFIG["LOOP_DELAY"])
                continue

            if not has_open_position(CONFIG["SYMBOL"]):
                signal = get_signal(rf_model, deep_model, df)
                if signal:
                    place_trade(signal, df)
                    last_trade_time = time.time()
                    send_telegram(f"📈 {signal.upper()} executed on {CONFIG['SYMBOL']}")
                else:
                    print("No valid signal.")
            else:
                print("Position already open.")

            time.sleep(CONFIG["LOOP_DELAY"])
        except KeyboardInterrupt:
            print("Stopping bot.")
            break
        except Exception as e:
            print("Error in loop:", e)
            time.sleep(CONFIG["LOOP_DELAY"])


if __name__ == "__main__":
    trade_loop()
