import pandas as pd

# load the raw DFW weather data
df = pd.read_csv("data/weather/dfw_weather_raw.csv")

# convert the DATE column into a datetime
df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce")

# pull out the temperature value from TMP
df["temp_c"] = (
    df["TMP"]
    .astype(str)
    .str.replace("+", "", regex=False)
    .str.split(",", expand=True)[0]
)

# convert the extracted temperature to a number
df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")

# remove NOAA's missing value code
df.loc[df["temp_c"] == 9999, "temp_c"] = None

# convert from tenths of a degree to °C
df["temp_c"] = df["temp_c"] / 10

# filter out temperatures that aren’t physically realistic
df = df[(df["temp_c"] > -50) & (df["temp_c"] < 60)]

# keep only the columns we care about and drop NaNs
df = df[["timestamp", "temp_c"]].dropna()

# average the data into hourly values
df = df.set_index("timestamp")
df_hourly = df.resample("1h").mean().reset_index()

# save the cleaned hourly weather data
df_hourly.to_csv(
    "data/weather/dfw_weather_hourly.csv",
    index=False
)

print("✅ DFW hourly weather cleaned (validated)")
print(df_hourly.describe())
print(df_hourly.head())
