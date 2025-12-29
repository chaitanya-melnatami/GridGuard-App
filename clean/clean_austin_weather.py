import pandas as pd

# load the raw NOAA weather data for Austin
df = pd.read_csv(
    "data/weather/72254013904.csv",
    low_memory=False
)

# turn the DATE column into a proper datetime
df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce")

# extract air temperature from TMP and convert it to °C
df["austin_temp_c"] = (
    df["TMP"]
    .astype(str)
    .str.split(",", expand=True)[0]
    .astype(float)
    / 10
)

# keep only the columns we actually need
df = df[["timestamp", "austin_temp_c"]]

# remove rows with missing data
df = df.dropna()

# filter out temperatures that don’t make physical sense
df = df[
    (df["austin_temp_c"] > -30) &
    (df["austin_temp_c"] < 60)
]

# resample the data into hourly averages
df = df.set_index("timestamp")

df_hourly = (
    df
    .resample("1h")
    .mean()
    .reset_index()
)

# save the cleaned hourly weather data to a CSV
df_hourly.to_csv(
    "data/weather/austin_weather_hourly.csv",
    index=False
)

print("✅ Austin hourly weather data cleaned and saved!")
print(df_hourly.head())
print(f"Rows: {len(df_hourly)}")
