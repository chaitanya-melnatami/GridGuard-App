import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FuncFormatter

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="GridGuard",
    layout="wide"
)

# -------------------------------------------------
# Header
# -------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("assets/logo.png", width=120)
with col2:
    st.title("GridGuard")
    st.caption("Multi‑Horizon Electricity Demand Forecasting for Grid Sustainability and Resilience")

st.divider()

# -------------------------------------------------
# Sidebar introduction
# -------------------------------------------------
st.sidebar.markdown("## About GridGuard")
st.sidebar.markdown(
    """
GridGuard helps estimate short‑term electricity demand so grid operators
can see stress coming before it becomes a problem.

It’s designed to support:
- Early blackout prevention
- Smarter use of renewables and storage
- Less reliance on emergency fossil‑fuel generators

The goal is simple: make the grid more reliable, cleaner, and easier to plan for.
"""
)

# -------------------------------------------------
# GitHub link
# -------------------------------------------------
st.sidebar.markdown(
    "[View the project on GitHub](https://github.com/chaitanya-melnatami/GridGuard)"
)
st.sidebar.divider()

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
page = st.sidebar.radio("Navigation", ["Live Forecast", "Model Validation"])
region = st.sidebar.selectbox("Region", ["Austin", "Dallas", "Houston"])
horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [1, 3, 6, 24],
    format_func=lambda x: f"{x}-Hour Ahead"
)
days_to_show = st.sidebar.slider("Days to Display (Validation)", 1, 30, 7)

# -------------------------------------------------
# Region configuration
# -------------------------------------------------
CONFIG = {
    "Austin": {
        "data": "data/merged/austin_scent_merged.csv",
        "temp_col": "austin_temp_c",
        "model": "models/austin_model"
    },
    "Dallas": {
        "data": "data/merged/dallas_ncent_merged.csv",
        "temp_col": "temp_c",
        "model": "models/dallas_model"
    },
    "Houston": {
        "data": "data/merged/houston_coast_merged.csv",
        "temp_col": "temp_c",
        "model": "models/houston_model"
    }
}
cfg = CONFIG[region]

# -------------------------------------------------
# Month mapping
# -------------------------------------------------
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# -------------------------------------------------
# Live forecast page
# -------------------------------------------------
if page == "Live Forecast":
    st.subheader("Live Demand Forecast")

    colA, colB = st.columns(2)

    with colA:
        demand = st.number_input(
            "Current Demand (MW)",
            min_value=1000.0,
            max_value=120000.0,
            value=15000.0,
            step=500.0
        )
        temp = st.number_input(
            "Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0
        )

    with colB:
        month_name = st.selectbox("Month", list(MONTHS.keys()))
        day = st.selectbox("Day", list(range(1, 32)))
        hour = st.slider("Hour of Day", 0, 23, 14)

    month = MONTHS[month_name]

    # -------------------------------------------------
    # Simple input sanity checks
    # -------------------------------------------------
    unrealistic = False
    warnings = []

    if temp >= 45:
        unrealistic = True
        warnings.append("Extreme temperature detected")
    if demand >= 60000:
        unrealistic = True
        warnings.append("Abnormally high demand input")

    if unrealistic:
        st.warning("Unusual inputs detected:")
        for w in warnings:
            st.write(f"- {w}")

    # -------------------------------------------------
    # Feature engineering
    # -------------------------------------------------
    day_of_year = int((month - 1) * 30.4 + day)

    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_doy = np.sin(2 * np.pi * day_of_year / 365)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365)

    X_live = pd.DataFrame([{
        "demand_mw": demand,
        cfg["temp_col"]: temp,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "dayofweek": 0,
        "is_weekend": 0,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy
    }])

    # -------------------------------------------------
    # Load model and predict
    # -------------------------------------------------
    with st.spinner("Running forecast model…"):
        time.sleep(0.4)
        model = joblib.load(f"{cfg['model']}_{horizon}h.pkl")
        prediction = model.predict(X_live)[0]

    st.metric(
        f"{horizon}-Hour Forecasted Demand",
        f"{prediction:,.0f} MW"
    )

    # -------------------------------------------------
    # Risk assessment logic
    # -------------------------------------------------
    delta_pct = ((prediction - demand) / demand) * 100

    if delta_pct < 5 and not unrealistic:
        risk = "🟢 Low Risk"
        reason = "Forecasted demand remains close to current levels."
    elif delta_pct < 12 or unrealistic:
        risk = "🟡 Medium Risk"
        reason = "Noticeable increase or unusual operating conditions."
    else:
        risk = "🔴 High Risk"
        reason = "Large projected surge may stress generation capacity."

    st.subheader("Grid Stress Risk")
    st.markdown(f"### {risk}")
    st.caption(f"Projected demand change: {delta_pct:.1f}%")
    st.info(f"Why: {reason}")

    # -------------------------------------------------
    # Projection chart with confidence range
    # -------------------------------------------------
    upper = prediction * 1.1
    lower = prediction * 0.9

    fig, ax = plt.subplots()
    ax.plot(["Now", f"+{horizon}h"], [demand, prediction], marker="o", linewidth=3)
    ax.fill_between(
        ["Now", f"+{horizon}h"],
        [demand, lower],
        [demand, upper],
        alpha=0.2,
        label="Confidence Range"
    )
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Future Demand Projection")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# -------------------------------------------------
# Model validation page
# -------------------------------------------------
else:
    st.subheader("Model Validation (Historical Performance)")

    df = pd.read_csv(cfg["data"], parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["dayofyear"] = df["timestamp"].dt.dayofyear

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df["target"] = df["demand_mw"].shift(-horizon)
    df = df.dropna()

    split = int(len(df) * 0.8)
    test = df.iloc[split:].copy()

    hours_to_show = days_to_show * 24
    test = test.tail(hours_to_show)

    FEATURES = [
        "demand_mw",
        cfg["temp_col"],
        "sin_hour",
        "cos_hour",
        "dayofweek",
        "is_weekend",
        "sin_doy",
        "cos_doy"
    ]

    model = joblib.load(f"{cfg['model']}_{horizon}h.pkl")
    pred = model.predict(test[FEATURES])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test["timestamp"], test["target"], label="Actual")
    ax.plot(test["timestamp"], pred, linestyle="--", label="Predicted")
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Time")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.info(
        "This view compares past model predictions against real demand data. "
        "It’s meant to show overall accuracy trends. "
        "Live decisions should be based on the Live Forecast page."
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption("GridGuard — Built for sustainability, resilience, and global welfare")
