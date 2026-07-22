import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("amd_cleaned.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df_feat = df.copy()

# RSI (Relative Strength Index) - measures market momentum
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df_feat["rsi_14"] = compute_rsi(df_feat["close"].shift(1), window=14)

# EMA - tracks price trend
df_feat["ema_12"] = df_feat["close"].shift(1).ewm(span=12, adjust=False).mean()
df_feat["ema_26"] = df_feat["close"].shift(1).ewm(span=26, adjust=False).mean()

# MACD - trend momentum
df_feat["macd"] = df_feat["ema_12"] - df_feat["ema_26"]
df_feat["macd_signal"] = df_feat["macd"].ewm(span=9, adjust=False).mean()
df_feat["macd_hist"] = df_feat["macd"] - df_feat["macd_signal"]

# Bollinger Bands - check volatility of price, in relation with recent changes
bb_window = 20
bb_mid = df_feat["close"].shift(1).rolling(bb_window).mean()
bb_std = df_feat["close"].shift(1).rolling(bb_window).std()
df_feat["bb_upper"] = bb_mid + 2 * bb_std
df_feat["bb_lower"] = bb_mid - 2 * bb_std
df_feat["bb_width"] = df_feat["bb_upper"] - df_feat["bb_lower"]
df_feat["bb_pct_b"] = (df_feat["close"].shift(1) - df_feat["bb_lower"]) / df_feat["bb_width"]

# ATR (Average True Range) - measures volatility of stock 
prev_close = df_feat["close"].shift(1)
high_low = df_feat["high"].shift(1) - df_feat["low"].shift(1)
high_prevclose = (df_feat["high"].shift(1) - prev_close).abs()
low_prevclose = (df_feat["low"].shift(1) - prev_close).abs()
true_range = pd.concat([high_low, high_prevclose, low_prevclose], axis=1).max(axis=1)
df_feat["atr_14"] = true_range.rolling(14).mean()

# HLC / OC Ratios - measures movement of stock throughout one day
df_feat["hlc_ratio"] = (df_feat["high"].shift(1) - df_feat["low"].shift(1)) / df_feat["close"].shift(1)
df_feat["oc_ratio"] = (df_feat["open"].shift(1) - df_feat["close"].shift(1)) / df_feat["close"].shift(1)

# ROC / Momentum - checks rate price changes
df_feat["roc_5"] = df_feat["close"].shift(1).pct_change(5)
df_feat["roc_10"] = df_feat["close"].shift(1).pct_change(10)
df_feat["momentum_5"] = df_feat["close"].shift(1) - df_feat["close"].shift(6)

# Ratio features - price/volume relative to their recent averages
df_feat["close_to_sma5"] = df_feat["close"].shift(1) / df_feat["sma_5"] - 1
df_feat["close_to_sma10"] = df_feat["close"].shift(1) / df_feat["sma_10"] - 1
df_feat["volume_to_volsma5"] = df_feat["volume"].shift(1) / df_feat["volume_sma_5"] - 1

# Return distribution
df_feat["return_skew_10"] = df_feat["return"].shift(1).rolling(10).skew()
df_feat["return_kurt_10"] = df_feat["return"].shift(1).rolling(10).kurt()

# Rolling min/max - looks at recent suport/resistance levels
df_feat["rolling_max_10"] = df_feat["high"].shift(1).rolling(10).max()
df_feat["rolling_min_10"] = df_feat["low"].shift(1).rolling(10).min()

# Calendar features
df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
df_feat["month"] = df_feat["date"].dt.month
df_feat["is_monday"] = (df_feat["day_of_week"] == 0).astype(int)
df_feat["is_month_end"] = df_feat["date"].dt.is_month_end.astype(int)

# Drop NaNs
before = len(df_feat)
df_feat = df_feat.dropna().reset_index(drop=True)
print(f"Dropped {before - len(df_feat)} rows due to feature warm-up periods")
print(f"Remaining rows: {len(df_feat)}")

# Correlation check
corr_with_target = df_feat.corr(numeric_only=True)["target"].sort_values(ascending=False)
print("\nTop correlated features with target:")
print(corr_with_target.head(15))

df_feat.to_csv("amd_features.csv", index=False)
print("Saved amd_features.csv")

# TEST / TRAIN / SPLIT
train_df, test_df = train_test_split(df_feat, test_size=0.2, shuffle=False)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\nTrain: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} rows)")
print(f"Test:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} rows)")

# ENCODING
categorical_cols = ["day_of_week", "month"]

train_df = pd.get_dummies(train_df, columns=categorical_cols, prefix=categorical_cols)
test_df = pd.get_dummies(test_df, columns=categorical_cols, prefix=categorical_cols)

train_cols = train_df.columns
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)
test_df = test_df[train_cols]

# FEATURE SCALING
no_scale_cols = ["date", "target", "is_monday", "is_month_end"]
no_scale_cols += [c for c in train_df.columns if c.startswith("day_of_week_") or c.startswith("month_")]
feature_cols = [c for c in train_df.columns if c not in no_scale_cols]

print(f"\nScaling {len(feature_cols)} numeric features")

scaler = StandardScaler()
train_scaled = train_df.copy()
test_scaled = test_df.copy()

train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

# Save df
train_scaled.to_csv("amd_train_scaled.csv", index=False)
test_scaled.to_csv("amd_test_scaled.csv", index=False)
print("Saved train file: amd_train_scaled.csv")
print("Saved test file: amd_test_scaled.csv")









