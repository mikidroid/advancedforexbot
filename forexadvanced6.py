# scalper_bot_full.py
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
import os
import math
import json

# ===========================
# CONFIG (scalper-tuned)
# ===========================
CONFIG = {
    "MODE": "scalper_safe",           # scalper_safe / scalper_aggressive
    "SYMBOL": "EURUSD",
    "PRIMARY_TF": mt5.TIMEFRAME_M1,   # scalping timeframe
    "HTF": mt5.TIMEFRAME_M5,          # higher timeframe for bias
    "BASE_FAST_SMA": 5,
    "BASE_SLOW_SMA": 20,
    "BASE_RSI": 9,
    "ATR_PERIOD": 7,
    "RISK_PER_TRADE": 0.004,          # 0.4% per trade (adjust as desired)
    "SL_ATR_MULT": 1.0,
    "TP_ATR_MULT": 1.5,
    "MODEL_RF_FILE": "rf_model.pkl",
    "MODEL_DEEP_FILE": "deep_model.h5",
    "SEQUENCE_LEN": 16,
    "MODEL_PROB_THRESHOLD": 0.25,
    "DRY_RUN": True,                  # start in dry-run; set False to trade live
    "LOOP_DELAY": 10,
    "MAGIC": 424242,
    "COOLDOWN_SECS": 120,
    "TELEGRAM_ENABLED": False,
    "TELEGRAM_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "SL_DOLLAR": 5,
    "USE_DOLLAR_SL": True,
    "TP_DOLLAR": 8,
    "USE_DOLLAR_TP": True,
    "RETRAIN_INTERVAL_HOURS": 2,
    "MIN_SAMPLES_RF": 20,
    "MIN_SAMPLES_DEEP": 50,
    "TRAINING_SAMPLES_FILE": "training_samples.csv",
    "TRADE_LOG_FILE": "trade_log.csv",
    "RETRAIN_ON_START": True,
    # news pause parameters (minutes)
    "NEWS_PAUSE_MINS_BEFORE": 30,
    "NEWS_PAUSE_MINS_AFTER": 30,
    # daily caps
    "DAILY_PROFIT_TARGET": 100.0,
    "DAILY_LOSS_LIMIT": -50.0,
    "DAILY_TRADE_LIMIT": 10,
    # lot caps (safety)
    "MIN_LOT": 0.01,
    "MAX_LOT": 0.10,
}

# FinancialModelingPrep API key (Option A)
NEWS_API_KEY = "y7XfDhqpKMIyJz9vK92ZupFJQsDGr3Ge"  # Put your key here; if blank, news check will be skipped
NEWS_CHECK_INTERVAL = 300  # cache news checks for 5 minutes

# ===========================
# DAILY LIMITS (CSV)
# ===========================
DAILY_LIMITS_FILE = "daily_trade_limits.csv"

def initialize_daily_limits():
    if not os.path.exists(DAILY_LIMITS_FILE):
        df = pd.DataFrame(columns=["date", "total_profit", "trade_count"])
        df.to_csv(DAILY_LIMITS_FILE, index=False)

def load_daily_limits():
    initialize_daily_limits()
    df = pd.read_csv(DAILY_LIMITS_FILE)
    today = dt.datetime.utcnow().date().isoformat()

    if not df.empty and df.iloc[-1]["date"] == today:
        return {
            "date": df.iloc[-1]["date"],
            "total_profit": float(df.iloc[-1]["total_profit"]),
            "trade_count": int(df.iloc[-1]["trade_count"])
        }

    limits = {"date": today, "total_profit": 0.0, "trade_count": 0}
    df = pd.concat([df, pd.DataFrame([limits])], ignore_index=True)
    df.to_csv(DAILY_LIMITS_FILE, index=False)
    return limits

def save_daily_limits(limits):
    df = pd.read_csv(DAILY_LIMITS_FILE)
    today = limits["date"]
    if today in df["date"].values:
        df.loc[df["date"] == today, ["total_profit", "trade_count"]] = [
            limits["total_profit"], limits["trade_count"]
        ]
    else:
        df = pd.concat([df, pd.DataFrame([limits])], ignore_index=True)
    df.to_csv(DAILY_LIMITS_FILE, index=False)

daily_limits = load_daily_limits()

def check_daily_limits():
    global daily_limits
    today = dt.datetime.utcnow().date().isoformat()
    if daily_limits["date"] != today:
        daily_limits = {"date": today, "total_profit": 0.0, "trade_count": 0}
        save_daily_limits(daily_limits)
    if daily_limits["total_profit"] >= CONFIG["DAILY_PROFIT_TARGET"]:
        print(f"🚫 Daily profit target reached (${CONFIG['DAILY_PROFIT_TARGET']}). Trading paused for today.")
        return False
    if daily_limits["total_profit"] <= CONFIG["DAILY_LOSS_LIMIT"]:
        print(f"🚫 Daily loss limit reached (${CONFIG['DAILY_LOSS_LIMIT']}). Trading paused for today.")
        return False
    if daily_limits["trade_count"] >= CONFIG["DAILY_TRADE_LIMIT"]:
        print(f"🚫 Daily trade limit reached ({CONFIG['DAILY_TRADE_LIMIT']} trades). Trading paused for today.")
        return False
    return True

def update_daily_limits(profit):
    global daily_limits
    daily_limits["total_profit"] += float(profit)
    daily_limits["trade_count"] += 1
    save_daily_limits(daily_limits)

# ===========================
# MODE TUNING
# ===========================
def apply_mode_settings():
    m = CONFIG["MODE"].lower()
    if m == "scalper_safe":
        CONFIG.update({"MODEL_PROB_THRESHOLD": 0.30, "RISK_PER_TRADE": 0.004, "SL_ATR_MULT": 1.0, "TP_ATR_MULT": 1.5})
    elif m == "scalper_aggressive":
        CONFIG.update({"MODEL_PROB_THRESHOLD": 0.20, "RISK_PER_TRADE": 0.006, "SL_ATR_MULT": 0.9, "TP_ATR_MULT": 1.4})
    else:
        CONFIG["MODE"] = "scalper_safe"
        apply_mode_settings()

apply_mode_settings()

# ===========================
# UTILITIES
# ===========================
def send_telegram(msg: str):
    if CONFIG.get("TELEGRAM_ENABLED", False) and CONFIG.get("TELEGRAM_TOKEN") and CONFIG.get("TELEGRAM_CHAT_ID"):
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
# MODELS LOADER / BUILDERS
# ===========================
def load_models():
    rf, deep = None, None
    if os.path.exists(CONFIG["MODEL_RF_FILE"]):
        try:
            rf = joblib.load(CONFIG["MODEL_RF_FILE"])
            print("✅ RF model loaded.")
        except Exception as e:
            print("⚠️ RF load error:", e)
            rf = None
    else:
        print("⚠️ RF model file not found:", CONFIG["MODEL_RF_FILE"])

    if os.path.exists(CONFIG["MODEL_DEEP_FILE"]):
        try:
            deep = load_model(CONFIG["MODEL_DEEP_FILE"], compile=False)
            print("✅ Deep model loaded.")
        except Exception as e:
            print("⚠️ Deep load error:", e)
            deep = None
    else:
        print("⚠️ Deep model file not found:", CONFIG["MODEL_DEEP_FILE"])

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
# FEATURES
# ===========================
def prepare_full_features(df_raw):
    df = df_raw.copy()
    df["ret1"] = df["close"].pct_change().fillna(0.0)
    if "tick_volume" in df.columns:
        df["vol"] = df["tick_volume"].astype(float).fillna(0.0)
    else:
        df["vol"] = 0.0
    fast = CONFIG["BASE_FAST_SMA"]
    slow = CONFIG["BASE_SLOW_SMA"]
    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()
    df["sma_diff"] = df["sma_fast"] - df["sma_slow"]
    df["sma_slope_fast"] = df["sma_fast"].diff()
    df["sma_slope_slow"] = df["sma_slow"].diff()
    df["rsi"] = compute_rsi(df["close"], CONFIG["BASE_RSI"])
    df["atr"] = calc_atr(df, CONFIG["ATR_PERIOD"])
    # ADX-ish
    h = df["high"]; l = df["low"]; c = df["close"]
    up_move = h.diff()
    down_move = l.diff() * -1
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(CONFIG["ATR_PERIOD"]).mean()
    plus_di = 100 * (plus_dm.rolling(CONFIG["ATR_PERIOD"]).sum() / (atr_series + 1e-9))
    minus_di = 100 * (minus_dm.rolling(CONFIG["ATR_PERIOD"]).sum() / (atr_series + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df["adx"] = dx.rolling(CONFIG["ATR_PERIOD"]).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# ===========================
# RF / DEEP helpers
# ===========================
def get_rf_expected_features(rf_model):
    if rf_model is None:
        return ["ret1", "sma_diff", "sma_slope_fast", "sma_slope_slow", "rsi", "atr", "adx", "vol"]
    try:
        if hasattr(rf_model, "feature_names_in_"):
            return list(rf_model.feature_names_in_)
    except Exception:
        pass
    return ["ret1", "sma_diff", "sma_slope_fast", "sma_slope_slow", "rsi", "atr", "adx", "vol"]

def align_rf_row(df, rf_model):
    expected = get_rf_expected_features(rf_model)
    for c in expected:
        if c not in df.columns:
            df[c] = 0.0
    return df[expected].iloc[[-1]].copy()

def prepare_deep_seq(df, deep_model):
    seq_len = CONFIG["SEQUENCE_LEN"]
    feature_cols = get_rf_expected_features(None)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df_seq = df[feature_cols].tail(seq_len).copy()
    if len(df_seq) < seq_len:
        pad_rows = seq_len - len(df_seq)
        top = pd.DataFrame(0.0, index=range(pad_rows), columns=df_seq.columns)
        df_seq = pd.concat([top, df_seq], ignore_index=True)
    seq = df_seq.values
    try:
        expected_feats = deep_model.input_shape[-1] if deep_model is not None else seq.shape[1]
        if seq.shape[1] < expected_feats:
            pad_width = expected_feats - seq.shape[1]
            seq = np.pad(seq, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
        elif seq.shape[1] > expected_feats:
            seq = seq[:, :expected_feats]
    except Exception:
        pass
    return seq

def ensemble_prediction_safe(rf_model, deep_model, df_features):
    rf_prob = 0.3
    deep_prob = 0.3
    if rf_model is not None:
        try:
            X_rf_row = align_rf_row(df_features, rf_model)
            rf_prob = float(rf_model.predict_proba(X_rf_row)[0][1])
        except Exception as e:
            print("RF predict error:", e)
            rf_prob = 0.3
    if deep_model is not None:
        try:
            seq = prepare_deep_seq(df_features, deep_model)
            seq_in = np.expand_dims(seq, axis=0)
            deep_prob = float(deep_model.predict(seq_in, verbose=0)[0][0])
        except Exception as e:
            print("Deep predict error:", e)
            deep_prob = 0.3
    w_rf, w_deep = 0.4, 0.6
    ensemble = w_rf * rf_prob + w_deep * deep_prob
    return rf_prob, deep_prob, ensemble

# ===========================
# SIGNAL DECISION
# ===========================
def get_signal(rf_model, deep_model, df_raw):
    df = prepare_full_features(df_raw)
    if df is None or df.empty:
        return None, None
    last = df.iloc[-1]
    primary_signal = None
    if last["sma_fast"] > last["sma_slow"] and last["rsi"] > 55:
        primary_signal = "buy"
    elif last["sma_fast"] < last["sma_slow"] and last["rsi"] < 45:
        primary_signal = "sell"
    if not primary_signal:
        return None, None
    rf_prob, deep_prob, ensemble = ensemble_prediction_safe(rf_model, deep_model, df)
    print(f"Signal={primary_signal} | rf={rf_prob:.3f} deep={deep_prob:.3f} ensemble={ensemble:.3f}")
    if ensemble >= CONFIG["MODEL_PROB_THRESHOLD"]:
        return primary_signal, ensemble
    else:
        print("Below model threshold; skipping trade.")
        return None, ensemble

# ===========================
# POSITION HELPERS
# ===========================
def has_open_position(symbol):
    try:
        pos = mt5.positions_get(symbol=symbol)
        return bool(pos)
    except Exception:
        return False

def current_position_direction(symbol):
    try:
        pos = mt5.positions_get(symbol=symbol)
        if pos:
            t = pos[0].type
            if t == mt5.POSITION_TYPE_BUY:
                return "buy"
            elif t == mt5.POSITION_TYPE_SELL:
                return "sell"
    except Exception:
        pass
    return None

# ===========================
# TRADE LOG / LABEL
# ===========================
def safe_read_csv(path):
    # Auto-create or repair empty file
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"⚠️ {path} missing or empty — creating fresh file.")
        with open(path, "w") as f:
            f.write("timestamp,open,high,low,close,volume,label,feature1,feature2,feature3\n")
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e} — resetting file.")
        with open(path, "w") as f:
            f.write("timestamp,open,high,low,close,volume,label,feature1,feature2,feature3\n")
        return pd.read_csv(path)
        
def append_trade_log(order_id, symbol, signal, entry_time_utc, entry_price, lot, sl, tp, features_row):
    base_cols = [
        "order_id", "symbol", "signal", "entry_time_utc",
        "entry_price", "lot", "sl", "tp", "status"
    ]
    feat_cols = list(features_row.keys())
    all_cols = base_cols + feat_cols
    df_row = pd.DataFrame([{
        **features_row,
        "order_id": int(order_id) if order_id else None,
        "symbol": symbol,
        "signal": signal,
        "entry_time_utc": entry_time_utc.isoformat(),
        "entry_price": entry_price,
        "lot": lot,
        "sl": sl,
        "tp": tp,
        "status": "open"
    }])[all_cols]
    file_path = CONFIG["TRADE_LOG_FILE"]
    if not os.path.exists(file_path):
        df_row.to_csv(file_path, index=False)
    else:
        existing = pd.read_csv(file_path)
        for col in all_cols:
            if col not in existing.columns:
                existing[col] = np.nan
        for col in existing.columns:
            if col not in all_cols:
                df_row[col] = np.nan
        df_row = df_row[existing.columns]
        df_row.to_csv(file_path, mode="a", header=False, index=False)

def update_trade_log_with_close(order_id, close_time_utc, profit):
    if not os.path.exists(CONFIG["TRADE_LOG_FILE"]):
        return
    df = pd.read_csv(CONFIG["TRADE_LOG_FILE"])
    idx = df.index[df["order_id"] == int(order_id)].tolist()
    if not idx:
        return
    i = idx[0]
    df.at[i, "status"] = "closed"
    df.at[i, "close_time_utc"] = close_time_utc.isoformat()
    df.at[i, "profit"] = profit
    df.to_csv(CONFIG["TRADE_LOG_FILE"], index=False)
    row = df.loc[[i]].copy()
    label = 1 if float(profit) > 0 else 0
    feat_cols = get_rf_expected_features(None)
    training_row = {}
    for c in feat_cols:
        training_row[c] = float(row.iloc[0].get(c, 0.0))
    training_row["label"] = label
    training_row["entry_time_utc"] = row.iloc[0]["entry_time_utc"]
    file_exists = os.path.exists(CONFIG["TRAINING_SAMPLES_FILE"])
    pd.DataFrame([training_row]).to_csv(CONFIG["TRAINING_SAMPLES_FILE"], mode="a", header=not file_exists, index=False)
    try:
        update_daily_limits(profit)
    except Exception as e:
        print("Daily limit update failed:", e)

def scan_closed_deals_and_label():
    if not os.path.exists(CONFIG["TRADE_LOG_FILE"]):
        return
    df = pd.read_csv(CONFIG["TRADE_LOG_FILE"])
    open_df = df[df["status"] == "open"]
    if open_df.empty:
        return
    try:
        for _, row in open_df.iterrows():
            order_id = int(row["order_id"]) if not pd.isna(row["order_id"]) else None
            entry_time = dt.datetime.fromisoformat(row["entry_time_utc"])
            deals = mt5.history_deals_get(entry_time, dt.datetime.utcnow())
            if deals is None:
                continue
            total_profit = 0.0
            matched = False
            for d in deals:
                try:
                    deal_order = getattr(d, "order", None) or getattr(d, "order_id", None) or getattr(d, "ticket", None)
                    deal_symbol = getattr(d, "symbol", "")
                    profit = float(getattr(d, "profit", 0.0))
                    if order_id is not None and deal_order == order_id and deal_symbol == row["symbol"]:
                        total_profit += profit
                        matched = True
                    elif order_id is None and deal_symbol == row["symbol"]:
                        deal_time = getattr(d, "time", None)
                        if deal_time:
                            dtdeal = dt.datetime.fromtimestamp(deal_time)
                            if abs((dtdeal - entry_time).total_seconds()) < 600:
                                total_profit += profit
                                matched = True
                except Exception:
                    continue
            if matched:
                update_trade_log_with_close(order_id, dt.datetime.utcnow(), total_profit)
                print(f"Labeled order {order_id} as closed profit={total_profit:.2f}")
    except Exception as e:
        print("Error scanning closed deals:", e)

# ===========================
# RETRAINING
# ===========================
def retrain_if_enough():
    path = CONFIG["TRAINING_SAMPLES_FILE"]
    if not os.path.exists(path):
        return
    df = safe_read_csv(path)
    if df.empty:
        return
    feat_cols = get_rf_expected_features(None)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    if len(df) >= CONFIG["MIN_SAMPLES_RF"]:
        try:
            from sklearn.ensemble import RandomForestClassifier
            X = df[feat_cols].values
            y = df["label"].values
            rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
            rf.fit(X, y)
            joblib.dump(rf, CONFIG["MODEL_RF_FILE"])
            print(f"Retrained & saved RF model on {len(df)} samples.")
        except Exception as e:
            print("RF retrain error:", e)
    if len(df) >= CONFIG["MIN_SAMPLES_DEEP"]:
        try:
            seq_len = CONFIG["SEQUENCE_LEN"]
            X_seq = []
            y_seq = []
            feats = df[feat_cols].values
            labels = df["label"].values
            for i in range(len(feats) - seq_len + 1):
                seq = feats[i:i+seq_len]
                lab = int(labels[i+seq_len-1])
                X_seq.append(seq)
                y_seq.append(lab)
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            deep = None
            if os.path.exists(CONFIG["MODEL_DEEP_FILE"]):
                try:
                    deep = load_model(CONFIG["MODEL_DEEP_FILE"], compile=False)
                    print("Loaded existing deep model for retrain.")
                except Exception:
                    deep = None
            if deep is None:
                deep = build_default_deep((seq_len, X_seq.shape[2]))
            es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            deep.fit(X_seq, y_seq, epochs=10, batch_size=64, validation_split=0.12, callbacks=[es], verbose=1)
            deep.save(CONFIG["MODEL_DEEP_FILE"])
            print(f"Retrained & saved Deep model on {len(X_seq)} sequences.")
        except Exception as e:
            print("Deep retrain error:", e)

# ===========================
# NEWS CHECK (FinancialModelingPrep)
# ===========================
_last_news_check = 0
_cached_news_pause = False

def parse_event_time(tstr):
    # attempt common ISO formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(tstr, fmt)
        except Exception:
            continue
    try:
        return dt.datetime.fromisoformat(tstr)
    except Exception:
        return None

def check_high_impact_news():
    """
    Returns True if there's a high-impact EUR or USD event within the configured before/after window.
    Caches result for NEWS_CHECK_INTERVAL seconds.
    """
    global _last_news_check, _cached_news_pause
    if not NEWS_API_KEY:
        # if API key not set, skip news check (assume safe)
        return False
    now = time.time()
    if now - _last_news_check < NEWS_CHECK_INTERVAL:
        return _cached_news_pause
    _last_news_check = now
    _cached_news_pause = False
    try:
        today = dt.datetime.utcnow().date().isoformat()
        url = f"https://financialmodelingprep.com/api/v4/economic_calendar?from={today}&to={today}&apikey={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()
        before = CONFIG.get("NEWS_PAUSE_MINS_BEFORE", 30)
        after = CONFIG.get("NEWS_PAUSE_MINS_AFTER", 30)
        for ev in data:
            # fields vary; try to be robust
            impact = ev.get("impact") or ev.get("importance") or ev.get("impact_name") or ""
            currency = ev.get("currency") or ev.get("country") or ""
            name = ev.get("event") or ev.get("title") or ev.get("name") or "event"
            date_str = ev.get("date") or ev.get("time") or ev.get("datetime") or ev.get("actualDate") or None
            if not date_str:
                continue
            event_time = parse_event_time(date_str)
            if event_time is None:
                continue
            # convert to naive UTC if timezone present
            if event_time.tzinfo is not None:
                event_time = event_time.astimezone(dt.timezone.utc).replace(tzinfo=None)
            mins_to_event = (event_time - dt.datetime.utcnow()).total_seconds() / 60.0
            # check only EUR or USD events (affects EUR/USD)
            if isinstance(currency, str) and any(x in currency.upper() for x in ["EUR", "USD"]):
                # consider only high-impact
                if isinstance(impact, str) and "HIGH" in impact.upper():
                    if -after <= mins_to_event <= before:
                        print(f"⚠️ High impact news: {name} ({currency}), in {mins_to_event:.1f} mins")
                        _cached_news_pause = True
                        break
            # some APIs put impact as integer or similar; be loose:
            if isinstance(impact, (int, float)) and impact >= 3:  # 3 often = high
                if -after <= mins_to_event <= before:
                    print(f"⚠️ High impact news (numeric): {name} ({currency}), in {mins_to_event:.1f} mins")
                    _cached_news_pause = True
                    break
    except Exception as e:
        print("News API error:", e)
        _cached_news_pause = False
    return _cached_news_pause

# ===========================
# PLACE TRADE
# ===========================
def compute_lot_from_risk(entry_price, sl_price, balance):
    # Compute per-lot loss in USD for EURUSD: per pip value on 1.0 lot ~ $10
    # We'll calculate ticks = abs(entry - sl) / point and per_lot_loss = ticks * pip_value(1.0)
    sym = mt5.symbol_info(CONFIG["SYMBOL"])
    point = sym.point if sym is not None else 0.0001
    pip_value_per_lot = 10.0  # approximate for EURUSD
    ticks = abs(entry_price - sl_price) / (point if point != 0 else 0.0001)
    per_lot_loss = (ticks / 10.0) * pip_value_per_lot  # ticks/10 => pips
    if per_lot_loss <= 0:
        return CONFIG["MIN_LOT"]
    risk_amt = balance * CONFIG["RISK_PER_TRADE"]
    raw_lot = risk_amt / per_lot_loss
    lot = round(raw_lot, 2)
    lot = max(CONFIG["MIN_LOT"], min(lot, CONFIG["MAX_LOT"]))
    return lot

def place_trade(signal, df_raw, rf_model=None, deep_model=None):
    symbol = CONFIG["SYMBOL"]
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick info, abort.")
        return None
    entry_price = float(tick.ask) if signal == "buy" else float(tick.bid)
    atr_series = calc_atr(df_raw, CONFIG["ATR_PERIOD"])
    if atr_series is None or atr_series.empty:
        print("No ATR available, skipping trade.")
        return None
    atr = float(atr_series.iloc[-1])

    # Compute SL price
    if CONFIG["USE_DOLLAR_SL"]:
        try:
            contract = 100000.0  # for standard contract
            price_dist = float(CONFIG["SL_DOLLAR"]) * entry_price / contract
            sl = entry_price - price_dist if signal == "buy" else entry_price + price_dist
        except Exception:
            sl = entry_price - CONFIG["SL_ATR_MULT"] * atr if signal == "buy" else entry_price + CONFIG["SL_ATR_MULT"] * atr
    else:
        sl = entry_price - CONFIG["SL_ATR_MULT"] * atr if signal == "buy" else entry_price + CONFIG["SL_ATR_MULT"] * atr

    # Compute TP price
    if CONFIG.get("USE_DOLLAR_TP", False):
        # we will compute pip distance then convert to price
        # assume pip_value per 0.01 lot ~ $0.10 -> for lot variable calculate later
        pass  # compute after lot size determined
    else:
        tp = entry_price + CONFIG["TP_ATR_MULT"] * atr if signal == "buy" else entry_price - CONFIG["TP_ATR_MULT"] * atr

    # account info
    info = mt5.account_info()
    if info is None:
        print("No account info.")
        return None
    balance = float(info.balance)

    # compute lot from risk
    lot = compute_lot_from_risk(entry_price, sl, balance)

    # compute TP if dollar-based
    if CONFIG.get("USE_DOLLAR_TP", False):
        pip_value = 10.0  # approx per 1.0 lot
        dollar_per_pip_per_lot = pip_value
        # dollar per pip for current lot:
        dollar_per_pip = dollar_per_pip_per_lot * lot
        pip_distance = CONFIG["TP_DOLLAR"] / (dollar_per_pip if dollar_per_pip != 0 else 1)
        price_per_pip = 0.0001
        tp_distance = pip_distance * price_per_pip
        tp = entry_price + tp_distance if signal == "buy" else entry_price - tp_distance

    if CONFIG["DRY_RUN"]:
        print(f"[DRY RUN] Would {signal.upper()} {symbol} @ {entry_price:.5f} SL={sl:.5f} TP={tp:.5f} lot={lot}")
        # log dry-run (no order_id)
        df_features = prepare_full_features(df_raw)
        feat_cols = get_rf_expected_features(None)
        features_row = {c: float(df_features.iloc[-1].get(c, 0.0)) for c in feat_cols}
        append_trade_log(None, symbol, signal, dt.datetime.utcnow(), entry_price, lot, sl, tp, features_row)
        return {"dry_run": True, "entry": entry_price, "sl": sl, "tp": tp, "lot": lot}

    order_type = mt5.ORDER_TYPE_BUY if signal == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": CONFIG["MAGIC"],
        "comment": f"{CONFIG['MODE']} scalper bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    print("Sending order:", request)
    try:
        result = mt5.order_send(request)
        print("OrderSendResult:", result)
        order_id = getattr(result, "order", None) or getattr(result, "order_id", None) or None
        df_features = prepare_full_features(df_raw)
        feat_cols = get_rf_expected_features(None)
        features_row = {c: float(df_features.iloc[-1].get(c, 0.0)) for c in feat_cols}
        append_trade_log(order_id, symbol, signal, dt.datetime.utcnow(), entry_price, lot, sl, tp, features_row)
        return result
    except Exception as e:
        print("order_send exception:", e)
        return None

# ===========================
# MAIN LOOP
# ===========================
def trade_loop():
    if not mt5.initialize():
        print("MT5 init failed.")
        return

    print("MT5 initialized. Loading models...")
    rf_model, deep_model = load_models()

    last_retrain_check = time.time() - (CONFIG["RETRAIN_INTERVAL_HOURS"] * 3600) if CONFIG.get("RETRAIN_ON_START", False) else time.time()
    last_trade_time = 0

    while True:
        try:
            scan_closed_deals_and_label()
            if time.time() - last_retrain_check >= CONFIG["RETRAIN_INTERVAL_HOURS"] * 3600:
                print("Checking for retrain...")
                retrain_if_enough()
                rf_model, deep_model = load_models()
                last_retrain_check = time.time()

            # news pause
            if check_high_impact_news():
                before = CONFIG.get("NEWS_PAUSE_MINS_BEFORE", 30)
                after = CONFIG.get("NEWS_PAUSE_MINS_AFTER", 30)
                pause = (before + after) * 60
                print(f"⏸️ High-impact news window active. Sleeping for {pause/60:.0f} minutes.")
                time.sleep(pause)
                continue

            df = get_data(CONFIG["SYMBOL"], CONFIG["PRIMARY_TF"], bars=500)
            if df is None or df.empty:
                print("No data; sleeping.")
                time.sleep(CONFIG["LOOP_DELAY"])
                continue

            signal, prob = get_signal(rf_model, deep_model, df)
            if signal:
                if not check_daily_limits():
                    print("Daily limit reached. Sleeping 5 minutes.")
                    time.sleep(300)
                    continue
                if has_open_position(CONFIG["SYMBOL"]):
                    print("Existing open position, skipping.")
                    time.sleep(CONFIG["LOOP_DELAY"])
                    continue
                dirn = current_position_direction(CONFIG["SYMBOL"])
                if dirn == signal:
                    print(f"Already in {dirn} position, skipping.")
                    time.sleep(CONFIG["LOOP_DELAY"])
                    continue
                if time.time() - last_trade_time < CONFIG["COOLDOWN_SECS"]:
                    print("Cooldown active, skipping.")
                    time.sleep(CONFIG["LOOP_DELAY"])
                    continue
                result = place_trade(signal, df, rf_model, deep_model)
                if result is not None:
                    # handle dry-run or real result
                    if isinstance(result, dict) and result.get("dry_run"):
                        print("Dry-run trade logged.")
                        last_trade_time = time.time()
                        send_telegram(f"[DRY] Trade simulated: {signal} {CONFIG['SYMBOL']} lot={result['lot']}")
                    else:
                        retcode = getattr(result, "retcode", None)
                        if retcode == mt5.TRADE_RETCODE_DONE or getattr(result, "retcode", None) == 10009:
                            print("Trade executed successfully.")
                            last_trade_time = time.time()
                            send_telegram(f"Trade executed: {signal} {CONFIG['SYMBOL']} prob={prob:.3f}")
                        else:
                            print("Trade result retcode:", retcode)
                else:
                    print("Trade not executed or failed.")
            time.sleep(CONFIG["LOOP_DELAY"])
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print("Main loop error:", e)
            time.sleep(5)

if __name__ == "__main__":
    print("Starting scalper bot MODE:", CONFIG["MODE"], "| DRY_RUN:", CONFIG["DRY_RUN"])
    print(">>> TEST ON DEMO ACCOUNT FIRST. Live trading involves risk.")
    trade_loop()
