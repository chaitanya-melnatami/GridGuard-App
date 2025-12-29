import pandas as pd

# load the ERCOT electricity demand data
df = pd.read_csv("data/ercot/ercot_demand.csv")

# convert the Hour Ending column into a datetime
df["timestamp"] = pd.to_datetime(df["Hour Ending"], errors="coerce")

# keep only the timestamp and COAST region data
df = df[["timestamp", "COAST"]]

# rename the demand column to be consistent
df = df.rename(columns={"COAST": "demand_mw"})

# remove rows with missing or invalid values
df = df.dropna()

# save the cleaned COAST region demand data
df.to_csv("data/ercot/ercot_coast_cleaned.csv", index=False)

print("✅ ERCOT COAST (Houston) cleaned!")
print(df.head())
