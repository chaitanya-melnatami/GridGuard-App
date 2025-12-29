import pandas as pd

# load the main ERCOT demand dataset
df = pd.read_csv("data/ercot/ercot_demand.csv")

# convert the Hour Ending column into a datetime
df["timestamp"] = pd.to_datetime(
    df["Hour Ending"],
    errors="coerce"
)

# keep only the timestamp and NCENT (DFW) region
df = df[["timestamp", "NCENT"]]

# rename the demand column to match the rest of the project
df = df.rename(columns={"NCENT": "demand_mw"})

# remove rows with missing or invalid data
df = df.dropna()

# save the cleaned NCENT demand data
df.to_csv("data/ercot/ercot_ncent_cleaned.csv", index=False)

print("✅ ERCOT NCENT (Dallas / DFW) cleaned!")
print(df.head())
