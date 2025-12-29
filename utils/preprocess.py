# utils/preprocess.py

def clean_input(df):
    required_cols = ["timestamp", "temperature", "demand_mw"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df
