import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

RAW_PATH = r"C:\Local_Git_Repository\mlai-\amd_stock_price.csv"  # change this to match where the file sits on your machine

df = pd.read_csv(RAW_PATH, skiprows=[1, 2], header=0)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
df.head()


print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

df.info()
df.isnull().sum().sort_values(ascending=False) #find the empty rows

#check for duplicate rows
duplicate_count = df.duplicated().sum()
print("Number of duplicated rows:", duplicate_count)

clean_df = df.copy()
clean_df.head(2)

#clean column names and convert types
clean_df.columns = clean_df.columns.str.strip().str.lower()
clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
clean_df = clean_df.sort_values("date").reset_index(drop=True)
clean_df.dtypes

#deal with missing values
missing = clean_df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

numeric_cols = ["close", "high", "low", "open", "volume"]

for col in numeric_cols:
    if clean_df[col].isnull().any():
        median_val = clean_df[col].median()
        clean_df[col] = clean_df[col].fillna(median_val)
        print(f"Filled {col} missing values with median: {median_val:.2f}")

print("\nRemaining missing values:")
print(clean_df[numeric_cols].isnull().sum())

#handle outliers

clean_df["return"] = clean_df["close"].pct_change()

q_low, q_high = clean_df["return"].quantile([0.001, 0.999])
n_extreme = ((clean_df["return"] < q_low) | (clean_df["return"] > q_high)).sum()
print(f"Extreme daily return outliers found: {n_extreme}")

clean_df["return"] = clean_df["return"].clip(q_low, q_high)

#create new columns for cleaner
for lag in [1, 2, 3, 5]:
    clean_df[f"close_lag{lag}"] = clean_df["close"].shift(lag)

clean_df["sma_5"] = clean_df["close"].shift(1).rolling(window=5).mean()
clean_df["sma_10"] = clean_df["close"].shift(1).rolling(window=10).mean()
clean_df["volatility_5"] = clean_df["return"].shift(1).rolling(window=5).std()
clean_df["volume_lag1"] = clean_df["volume"].shift(1)
clean_df["volume_sma_5"] = clean_df["volume"].shift(1).rolling(window=5).mean()
clean_df["hl_range_lag1"] = clean_df["high"].shift(1) - clean_df["low"].shift(1)

# Target: next trading day's closing price
clean_df["target"] = clean_df["close"].shift(-1)

clean_df.tail()

#remove structural missing values
before = len(clean_df)
clean_df = clean_df.dropna().reset_index(drop=True)
after = len(clean_df)
print(f"Dropped {before - after} rows with structural NaNs (feature warm-up period + final row with no target)")
print(f"Remaining rows: {after}")

clean_df.to_csv("amd_cleaned.csv", index=False)
print("Saved cleaned file: amd_cleaned.csv")