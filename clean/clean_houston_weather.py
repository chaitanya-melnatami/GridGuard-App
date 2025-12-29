import pandas as pd

# load the raw Houston weather data
df = pd.read_csv("data/weather/houston_weather_raw.csv", low_memory=False)

# convert the DATE column into a datetime
df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce")

# extract the air temperature from TMP and convert to °C
df["temp_c"] = (
    df["TMP"]
    .str.split(",", expand=True)[0]
    .astype(float) / 10
)

# filter out temperatures that don’t make sense
df = df[(df["temp_c"] > -40) & (df["temp_c"] < 50)]

# average the temperature data by hour
df = (
    df[["timestamp", "temp_c"]]
    .set_index("timestamp")
    .resample("1h")
    .mean()
    .reset_index()
)

# save the cleaned hourly Houston weather data
df.to_csv("data/weather/houston_weather_hourly.csv", index=False)

print("✅ Houston hourly weather cleaned!")
print(df.describe())
