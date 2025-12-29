import pandas as pd

# load the cleaned ERCOT NCENT demand data
ercot = pd.read_csv(
    "data/ercot/ercot_ncent_cleaned.csv",
    parse_dates=["timestamp"]
)

# load the cleaned DFW hourly weather data
weather = pd.read_csv(
    "data/weather/dfw_weather_hourly.csv",
    parse_dates=["timestamp"]
)

# merge the demand and weather data using the timestamp
df = pd.merge(
    ercot,
    weather,
    on="timestamp",
    how="inner"
)

# sort the data by time so it’s ready for modeling
df = df.sort_values("timestamp")

# save the final merged Dallas dataset
df.to_csv(
    "data/merged/dallas_ncent_merged.csv",
    index=False
)

print("✅ Dallas (NCENT) merged dataset created!")
print(df.head())
print("Rows:", len(df))
