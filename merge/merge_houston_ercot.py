import pandas as pd
import os

# make sure the merged data folder exists
os.makedirs("data/merged", exist_ok=True)

# load the cleaned ERCOT COAST demand data
ercot = pd.read_csv(
    "data/ercot/ercot_coast_cleaned.csv",
    parse_dates=["timestamp"]
)

# load the cleaned Houston hourly weather data
weather = pd.read_csv(
    "data/weather/houston_weather_hourly.csv",
    parse_dates=["timestamp"]
)

# merge demand and weather data on timestamp and sort by time
df = pd.merge(
    ercot,
    weather,
    on="timestamp",
    how="inner"
).sort_values("timestamp")

# save the final merged Houston dataset
df.to_csv("data/merged/houston_coast_merged.csv", index=False)

print("✅ Houston (COAST) merged!")
print(df.head())
print("Rows:", len(df))
