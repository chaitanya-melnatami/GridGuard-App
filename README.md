GridGuard
GridGuard is a Streamlit web app for short‑term electricity demand forecasting.
It’s designed to help understand grid stress risks under different conditions.

What it does
Predicts electricity demand 1, 3, 6, or 24 hours ahead
Supports multiple regions (Austin, Dallas, Houston)
Flags potentially risky or unrealistic inputs
Shows simple confidence ranges and risk explanations
Includes historical model validation with adjustable time windows

Why it matters
Electricity grids are sensitive to spikes in demand, weather extremes, and timing.
GridGuard helps visualize how demand may change and when the grid could be stressed.

How it works
Uses trained machine learning models per region and forecast horizon
Applies basic feature engineering (time of day, seasonality, temperature)
Runs entirely in a single app.py Streamlit app

Tech stack
Python
Streamlit
Pandas / NumPy
Matplotlib
Scikit‑learn (via joblib models)

Run locally
pip install -r requirements.txt
streamlit run app.py

Notes
This project is for educational and analytical purposes, not real‑time grid operations.