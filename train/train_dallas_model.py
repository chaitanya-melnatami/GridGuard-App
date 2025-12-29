import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/merged/dallas_ncent_merged.csv"
TEMP_COL = "temp_c"
HORIZONS = [1, 3, 6, 24]

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["dayofyear"] = df["timestamp"].dt.dayofyear

df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

FEATURES = [
    "demand_mw",
    TEMP_COL,
    "sin_hour",
    "cos_hour",
    "dayofweek",
    "is_weekend",
    "sin_doy",
    "cos_doy",
]

for H in HORIZONS:
    df_h = df.copy()
    df_h["target"] = df_h["demand_mw"].shift(-H)
    df_h = df_h.dropna()

    X = df_h[FEATURES]
    y = df_h["target"]

    split = int(len(df_h) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    out_path = f"models/dallas_model_{H}h.pkl"
    joblib.dump(model, out_path)

    print(f"✅ Dallas {H}-Hour model trained")
    print(f"MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW")
