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
    "DRY_RUN": False,                    # start in dry-run (change to False to trade)
    "LOOP_DELAY": 20,
    "MAGIC": 424242,
    "COOLDOWN_SECS": 300,
    "TELEGRAM_ENABLED": False,
    "TELEGRAM_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "SL_DOLLAR": 15,
    "USE_DOLLAR_SL": True,
    "TP_DOLLAR": 10,  # desired profit per trade in USD
    "USE_DOLLAR_TP": True,  # set True to activate dollar-based TP

    # Self-learning / retrain config:
    "RETRAIN_INTERVAL_HOURS": 1,
    "MIN_SAMPLES_RF": 10,        # min labeled trades to retrain RF
    "MIN_SAMPLES_DEEP": 30,     # min labeled trades to retrain deep
    "TRAINING_SAMPLES_FILE": "training_samples.csv",
    "TRADE_LOG_FILE": "trade_log.csv",  # logs entries until closed & labeled
    "RETRAIN_ON_START": True,   # check for retrain on startup
}

# Mode tuning
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
# Models loader / default builders
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
# Feature engineering (comprehensive)
# ===========================
def prepare_full_features(df_raw):
    """Return df with engineered features used by models (keeps original OHLC)."""
    df = df_raw.copy()
    # returns
    df["ret1"] = df["close"].pct_change().fillna(0.0)
    # volumes (tick volume)
    if "tick_volume" in df.columns:
        df["vol"] = df["tick_volume"].astype(float).fillna(0.0)
    else:
        df["vol"] = 0.0
    # SMAs
    fast = CONFIG["BASE_FAST_SMA"]
    slow = CONFIG["BASE_SLOW_SMA"]
    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()
    # diffs & slopes
    df["sma_diff"] = df["sma_fast"] - df["sma_slow"]
    df["sma_slope_fast"] = df["sma_fast"].diff()
    df["sma_slope_slow"] = df["sma_slow"].diff()
    # rsi & atr
    df["rsi"] = compute_rsi(df["close"], CONFIG["BASE_RSI"])
    df["atr"] = calc_atr(df, CONFIG["ATR_PERIOD"])
    # ADX approximation
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
# Feature alignment helpers
# ===========================
def get_rf_expected_features(rf_model):
    """Return list of expected RF features if available, otherwise a sensible default."""
    if rf_model is None:
        # default feature order we use for training compatibility
        return ["ret1", "sma_diff", "sma_slope_fast", "sma_slope_slow", "rsi", "atr", "adx", "vol"]
    try:
        if hasattr(rf_model, "feature_names_in_"):
            return list(rf_model.feature_names_in_)
    except Exception:
        pass
    # fallback
    return ["ret1", "sma_diff", "sma_slope_fast", "sma_slope_slow", "rsi", "atr", "adx", "vol"]

def align_rf_row(df, rf_model):
    """Return single-row DataFrame aligned to RF expected features (order + missing columns)."""
    expected = get_rf_expected_features(rf_model)
    # ensure columns exist in df; if not, add neutral 0
    for c in expected:
        if c not in df.columns:
            df[c] = 0.0
    # select last row and reorder
    return df[expected].iloc[[-1]].copy()

def prepare_deep_seq(df, deep_model):
    """Return numpy array shape (seq_len, n_features) matching deep model input last axis.
       Pads columns with zeros if deep expects more features than available."""
    seq_len = CONFIG["SEQUENCE_LEN"]
    # use a default feature set - same as RF expected features
    feature_cols = get_rf_expected_features(None)
    # Build df_seq (last seq_len rows) ensuring these columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df_seq = df[feature_cols].tail(seq_len).copy()
    if len(df_seq) < seq_len:
        # pad at top with zeros to reach seq_len
        pad_rows = seq_len - len(df_seq)
        top = pd.DataFrame(0.0, index=range(pad_rows), columns=df_seq.columns)
        df_seq = pd.concat([top, df_seq], ignore_index=True)
    seq = df_seq.values  # shape (seq_len, n_features_current)
    # deep model expected feature dimension
    try:
        expected_feats = deep_model.input_shape[-1] if deep_model is not None else seq.shape[1]
        if seq.shape[1] < expected_feats:
            pad_width = expected_feats - seq.shape[1]
            seq = np.pad(seq, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
        elif seq.shape[1] > expected_feats:
            # truncate extra columns on right if too many
            seq = seq[:, :expected_feats]
    except Exception:
        expected_feats = seq.shape[1]
    return seq  # shape (seq_len, expected_feats)

# ===========================
# Ensemble prediction (safe)
# ===========================
def ensemble_prediction_safe(rf_model, deep_model, df_features):
    rf_prob = 0.3
    deep_prob = 0.3

    # RF
    if rf_model is not None:
        try:
            X_rf_row = align_rf_row(df_features, rf_model)  # DataFrame
            rf_prob = float(rf_model.predict_proba(X_rf_row)[0][1])
        except Exception as e:
            print("RF predict error:", e)
            rf_prob = 0.3

    # Deep
    if deep_model is not None:
        try:
            seq = prepare_deep_seq(df_features, deep_model)  # (seq_len, n_feats)
            seq_in = np.expand_dims(seq, axis=0)  # (1, seq_len, n_feats)
            deep_prob = float(deep_model.predict(seq_in, verbose=0)[0][0])
        except Exception as e:
            print("Deep predict error:", e)
            deep_prob = 0.3

    # Weighted ensemble
    w_rf, w_deep = 0.4, 0.6
    ensemble = w_rf * rf_prob + w_deep * deep_prob
    return rf_prob, deep_prob, ensemble

# ===========================
# Signal decision
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
# Position helpers
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
# Trade logging & labeling helpers
# ===========================
def append_trade_log(order_id, symbol, signal, entry_time_utc, entry_price, lot, sl, tp, features_row):
    """Append an open trade entry to trade_log.csv"""
    cols = ["order_id", "symbol", "signal", "entry_time_utc", "entry_price", "lot", "sl", "tp", "status"] + list(features_row.keys())
    file_exists = os.path.exists(CONFIG["TRADE_LOG_FILE"])
    row = {k: v for k, v in features_row.items()}
    row.update({
        "order_id": int(order_id) if order_id is not None else None,
        "symbol": symbol,
        "signal": signal,
        "entry_time_utc": entry_time_utc.isoformat(),
        "entry_price": entry_price,
        "lot": lot,
        "sl": sl,
        "tp": tp,
        "status": "open"
    })
    df_row = pd.DataFrame([row])
    if not file_exists:
        df_row.to_csv(CONFIG["TRADE_LOG_FILE"], index=False)
    else:
        df_row.to_csv(CONFIG["TRADE_LOG_FILE"], mode="a", header=False, index=False)

def update_trade_log_with_close(order_id, close_time_utc, profit):
    """Mark trade as closed in trade_log.csv and append labeled sample to training_samples.csv"""
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

    # Move row to training_samples.csv with label
    row = df.loc[[i]].copy()
    # Build label (1 win, 0 loss/zero)
    label = 1 if float(profit) > 0 else 0
    # keep only feature columns + label and timestamp
    # features defined by get_rf_expected_features(None)
    feat_cols = get_rf_expected_features(None)
    training_row = {}
    for c in feat_cols:
        training_row[c] = float(row.iloc[0].get(c, 0.0))
    training_row["label"] = label
    training_row["entry_time_utc"] = row.iloc[0]["entry_time_utc"]
    # append
    file_exists = os.path.exists(CONFIG["TRAINING_SAMPLES_FILE"])
    pd.DataFrame([training_row]).to_csv(CONFIG["TRAINING_SAMPLES_FILE"], mode="a", header=not file_exists, index=False)

def scan_closed_deals_and_label():
    """Check history for closed deals that match logged open trades. If found, label them."""
    if not os.path.exists(CONFIG["TRADE_LOG_FILE"]):
        return
    df = pd.read_csv(CONFIG["TRADE_LOG_FILE"])
    open_df = df[df["status"] == "open"]
    if open_df.empty:
        return

    # lookback window: from earliest open entry_time to now
    try:
        for _, row in open_df.iterrows():
            order_id = int(row["order_id"]) if not pd.isna(row["order_id"]) else None
            entry_time = dt.datetime.fromisoformat(row["entry_time_utc"])
            # query deals from entry_time to now
            deals = mt5.history_deals_get(entry_time, dt.datetime.utcnow())
            if deals is None:
                continue
            # find deal with matching order id (or symbol and near entry price)
            total_profit = 0.0
            matched = False
            for d in deals:
                try:
                    # each deal has .order or .order_id attribute depending on build; check both
                    deal_order = getattr(d, "order", None) or getattr(d, "order_id", None) or getattr(d, "ticket", None)
                    deal_symbol = getattr(d, "symbol", "")
                    profit = float(getattr(d, "profit", 0.0))
                    if order_id is not None and deal_order == order_id and deal_symbol == row["symbol"]:
                        total_profit += profit
                        matched = True
                    # fallback: if order id missing, match by symbol and time proximity and price
                    elif order_id is None and deal_symbol == row["symbol"]:
                        # time check: deal.time is seconds since epoch
                        deal_time = getattr(d, "time", None)
                        if deal_time:
                            dtdeal = dt.datetime.fromtimestamp(deal_time)
                            # if within 10 minutes after entry, consider it (heuristic)
                            if abs((dtdeal - entry_time).total_seconds()) < 600:
                                total_profit += profit
                                matched = True
                except Exception:
                    continue
            if matched:
                # update log & create labeled sample
                update_trade_log_with_close(order_id, dt.datetime.utcnow(), total_profit)
                print(f"Labeled order {order_id} as closed profit={total_profit:.2f}")
    except Exception as e:
        print("Error scanning closed deals:", e)

# ===========================
# Retraining
# ===========================
def retrain_if_enough():
    """Retrain RF and/or Deep models if enough labeled samples are available."""
    # require training_samples.csv to exist
    path = CONFIG["TRAINING_SAMPLES_FILE"]
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    feat_cols = get_rf_expected_features(None)
    # ensure feature columns exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    # RF retrain
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

    # Deep retrain (sequence)
    if len(df) >= CONFIG["MIN_SAMPLES_DEEP"]:
        try:
            # create sequences by sliding over training rows
            seq_len = CONFIG["SEQUENCE_LEN"]
            X_seq = []
            y_seq = []
            feats = df[feat_cols].values
            labels = df["label"].values
            # Build sequences: label of last item in sequence
            for i in range(len(feats) - seq_len + 1):
                seq = feats[i:i+seq_len]
                lab = int(labels[i+seq_len-1])
                X_seq.append(seq)
                y_seq.append(lab)
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            # build or load model
            deep = None
            if os.path.exists(CONFIG["MODEL_DEEP_FILE"]):
                try:
                    deep = load_model(CONFIG["MODEL_DEEP_FILE"], compile=False)
                    print("Loaded existing deep model for retrain.")
                except Exception:
                    deep = None
            if deep is None:
                deep = build_default_deep((seq_len, X_seq.shape[2]))
            # train
            es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            deep.fit(X_seq, y_seq, epochs=10, batch_size=64, validation_split=0.12, callbacks=[es], verbose=1)
            deep.save(CONFIG["MODEL_DEEP_FILE"])
            print(f"Retrained & saved Deep model on {len(X_seq)} sequences.")
        except Exception as e:
            print("Deep retrain error:", e)

# ===========================
# Place trade (safe)
# ===========================
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

    # compute SL & TP (dollar or ATR)
    if CONFIG["USE_DOLLAR_TP"]:
        pip_value = 10
        lot = 0.1
        pip_dist = CONFIG["TP_DOLLAR"] / (pip_value * lot)
        tp = entry_price + pip_dist * 0.0001 if signal == "buy" else entry_price - pip_dist * 0.0001
    else:
        tp = entry_price + CONFIG["TP_ATR_MULT"] * atr if signal == "buy" else entry_price - CONFIG["TP_ATR_MULT"] * atr
    
    # Compute SL using dollar amount (converted to price distance) or ATR-based SL
    if CONFIG["USE_DOLLAR_SL"]:
        try:
            sym_info = mt5.symbol_info(symbol)
            contract = float(getattr(sym_info, "trade_contract_size", 100000) or 100000)
            # avoid division by zero
            if entry_price == 0 or contract == 0:
                raise Exception("Invalid entry_price/contract for dollar SL calculation.")
            # per_lot_loss = abs(entry_price - sl) * contract / entry_price
            # => abs(entry_price - sl) = SL_DOLLAR * entry_price / contract
            price_dist = float(CONFIG["SL_DOLLAR"]) * entry_price / contract
            sl = entry_price - price_dist if signal == "buy" else entry_price + price_dist
        except Exception as e:
            print("Dollar SL calc error:", e)
            # fallback to ATR-based SL if anything goes wrong
            sl = entry_price - CONFIG["SL_ATR_MULT"] * atr if signal == "buy" else entry_price + CONFIG["SL_ATR_MULT"] * atr
    else:
        sl = entry_price - CONFIG["SL_ATR_MULT"] * atr if signal == "buy" else entry_price + CONFIG["SL_ATR_MULT"] * atr


    # compute lot (approx)
    info = mt5.account_info()
    if info is None:
        print("No account info.")
        return None
    balance = float(info.balance)
    risk_amt = balance * CONFIG["RISK_PER_TRADE"]

    sym = mt5.symbol_info(symbol)
    if sym is not None and getattr(sym, "trade_contract_size", None) is not None:
        point = sym.point
        contract = getattr(sym, "trade_contract_size", 100000)
        ticks = abs(entry_price - sl) / (point if point != 0 else 0.0001)
        per_lot_loss = ticks * ((point / entry_price) * contract)
    else:
        per_lot_loss = abs(entry_price - sl) * 100000 * 0.0001 / (entry_price if entry_price != 0 else 1)

    if per_lot_loss <= 0:
        print("Per-lot loss <= 0, abort.")
        return None

    raw_lot = risk_amt / per_lot_loss
    lot = round(raw_lot, 2)
    lot = max(0.01, min(lot, 1.0))

    # compute TP either dollar-based or ATR-based
    if CONFIG.get("USE_DOLLAR_TP", False):
        pip_value = 10.0 if "USD" in symbol else 9.0
        dollar_per_pip = pip_value * lot
        pip_distance = CONFIG["TP_DOLLAR"] / (dollar_per_pip if dollar_per_pip != 0 else 1)
        price_per_pip = 0.0001
        tp_distance = pip_distance * price_per_pip
        tp = entry_price + tp_distance if signal == "buy" else entry_price - tp_distance
    else:
        tp = entry_price + CONFIG["TP_ATR_MULT"] * atr if signal == "buy" else entry_price - CONFIG["TP_ATR_MULT"] * atr

    if CONFIG["DRY_RUN"]:
        print(f"[DRY RUN] Would {signal.upper()} {symbol} @ {entry_price:.5f} SL={sl:.5f} TP={tp:.5f} lot={lot}")
        # still log dry-run as not executed (no order_id)
        # optionally we could log simulation results
        return None


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
        "comment": f"{CONFIG['MODE']} AI bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    print("Sending order:", request)
    try:
        result = mt5.order_send(request)
        print("OrderSendResult:", result)
        # log order if created (order id)
        order_id = getattr(result, "order", None) or getattr(result, "order_id", None) or None
        # append features (align row) for training later
        df_features = prepare_full_features(df_raw)
        # align features row using RF expected features
        feat_cols = get_rf_expected_features(None)
        features_row = {}
        last_row = df_features.iloc[-1]
        for c in feat_cols:
            features_row[c] = float(last_row.get(c, 0.0))
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
        # periodic scan for closed deals -> label trades
        scan_closed_deals_and_label()

        # periodic retrain check
        if time.time() - last_retrain_check >= CONFIG["RETRAIN_INTERVAL_HOURS"] * 3600:
            print("Checking for retrain...")
            retrain_if_enough()
            # after retrain attempt, reload models
            rf_model, deep_model = load_models()
            last_retrain_check = time.time()

        df = get_data(CONFIG["SYMBOL"], CONFIG["PRIMARY_TF"])
        if df is None or df.empty:
            print("No data; sleeping.")
            time.sleep(CONFIG["LOOP_DELAY"])
            continue

        signal, prob = get_signal(rf_model, deep_model, df)
        if signal:
            # skip if position exists
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
            if result is not None and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                last_trade_time = time.time()
                send_telegram(f"Trade executed: {signal} {CONFIG['SYMBOL']} prob={prob:.3f}")
            else:
                print("Trade not executed or failed.")
        else:
            # no signal
            pass

        time.sleep(CONFIG["LOOP_DELAY"])

# ===========================
# Run
# ===========================
if __name__ == "__main__":
    print("Starting bot in MODE:", CONFIG["MODE"], "| DRY_RUN:", CONFIG["DRY_RUN"])
    print("Use DEMO first. WARNING: Live trading risks capital.")
    trade_loop()
