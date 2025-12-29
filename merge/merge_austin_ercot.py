import pandas as pd

# load the cleaned Austin weather data
weather = pd.read_csv(
    "data/weather/austin_weather_hourly.csv",
    parse_dates=["timestamp"]
)

# load the cleaned ERCOT SCENT demand data
ercot = pd.read_csv(
    "data/ercot/ercot_scent_cleaned.csv",
    parse_dates=["timestamp"]
)

# merge weather and demand data on timestamp
merged = pd.merge(
    ercot,
    weather,
    on="timestamp",
    how="inner"
)

# save the final merged dataset
merged.to_csv(
    "data/ercot/austin_scent_merged.csv",
    index=False
)

print("✅ Final merged dataset created!")
print(merged.head())
print(f"Rows: {len(merged)}")
