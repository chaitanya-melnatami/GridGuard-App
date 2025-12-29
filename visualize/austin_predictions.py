import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# CONFIG
HORIZON_HOURS = 1   # change to 3, 6, or 24
DATA_PATH = "data/merged/austin_scent_merged.csv"
MODEL_PATH = f"models/austin_model_{HORIZON_HOURS}h.pkl"

# LOAD DATA
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# time features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# target (multi-hour ahead)
df["target_demand"] = df["demand_mw"].shift(-HORIZON_HOURS)
df = df.dropna()

# features for prediction
features = [
    "demand_mw",
    "austin_temp_c",
    "hour",
    "dayofweek",
    "is_weekend"
]

X = df[features]
y = df["target_demand"]

# time-based train-test split
split_index = int(len(df) * 0.8)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]
timestamps = df["timestamp"].iloc[split_index:]

# LOAD MODEL & PREDICT
model = joblib.load(MODEL_PATH)
y_pred = model.predict(X_test)

# PLOT (first 200 hours)
plt.figure(figsize=(12, 5))
plt.plot(timestamps[:200], y_test[:200], label="Actual Demand", linewidth=2)
plt.plot(timestamps[:200], y_pred[:200], label="Predicted Demand", linestyle="--")
plt.title(f"Austin Electricity Demand Forecast ({HORIZON_HOURS}-Hour Ahead)")
plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
