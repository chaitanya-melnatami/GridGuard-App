import pandas as pd

# load the ERCOT electricity demand data
df = pd.read_csv("data/ercot/ercot_demand.csv")

# clean and convert the Hour Ending column into a datetime
# this assumes the date is already included and just removes "HE "
df["timestamp"] = pd.to_datetime(
    df["Hour Ending"].str.replace("HE ", "", regex=False),
    errors="coerce"
)

# keep only the timestamp and SCENT region data
df = df[["timestamp", "SCENT"]]

# rename the demand column to be consistent
df = df.rename(columns={"SCENT": "demand_mw"})

# drop rows with missing or invalid values
df = df.dropna()

# save the cleaned SCENT demand data
df.to_csv("data/ercot/ercot_scent_cleaned.csv", index=False)

print("✅ ERCOT SCENT cleaned!")
print(df.head())
