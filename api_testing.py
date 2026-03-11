from entsoe import *
import pandas as pd
import json
import math
import numpy as np
from dotenv import load_dotenv
import os

# load entsoe API_TOKEN
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

# load carbon/water intensities (.json)
# assuming intensities values to be per kWh 

with open('intensities.json', 'r') as f:
    data = json.load(f)

# extract carbon and water intensities from the JSON file
carbon = data['carbon']
water = data['water']

# define dictionaries for carbon and water intensities
carbon_intensities = {
    "Biomass": carbon["biomass-ipcc-2014"],
    "Fossil Gas": carbon["gas-ngcc-unece-2020"],
    "Fossil Hard coal": 0.95 * carbon["coal-pc-unece-2020"] + 0.05 * carbon["coal-sc-unece-2020"],
    "Fossil Brown coal/Lignite": 0.95 * carbon["coal-pc-unece-2020"] + 0.05 * carbon["coal-sc-unece-2020"],
    "Fossil Coal-derived gas": carbon["coal-igcc-unece-2020"],
    "Hydro Run-of-river and poundage": 0.95 * carbon["hydro-medium-unece-2020"] + 0.05 * carbon["hydro-large-unece-2020"],
    "Hydro Water Reservoir": 0.95 * carbon["hydro-medium-unece-2020"] + 0.05 * carbon["hydro-large-unece-2020"],
    "Solar": 0.45 * carbon["solar-pv-poly-si-roof-unece-2020"] + 0.45 * carbon["solar-pv-poly-si-ground-unece-2020"] + 
             0.025 * (carbon["solar-pv-cdte-ground-unece-2020"] + carbon["solar-pv-cdte-roof-unece-2020"] + 
                      carbon["solar-pv-cigs-ground-unece-2020"] + carbon["solar-pv-poly-si-roof-unece-2020"]),
    "Wind Offshore": 0.5 * carbon["wind-offshore-concrete-unece-2020"] + 0.5 * carbon["wind-offshore-steel-unece-2020"],
    "Wind Onshore": carbon["wind-onshore-unece-2020"],
    "Nuclear": carbon["nuclear-unece-2020"],
    "Geothermal": carbon["geothermal-ipcc-2014"],
    "Waste": 0.0,
    "Fossil Oil": 0.0,
    "Hydro Pumped Storage": 0.0
}

water_intensities = {
    "Biomass": 0.25 * (water["biopower-biogas-tower"] + water["biopower-steam-once-through"] + 
                       water["biopower-steam-pond"] + water["biopower-steam-tower"]),
    "Fossil Gas": 0.33 * (water["gas-ngcc-tower"] + water["gas-ngcc-once-through"] + water["gas-ngcc-pond"]),
    "Fossil Hard coal": 0.3 * (water["coal-pc-subc-tower"] + water["coal-pc-subc-once-through"] + water["coal-pc-subc-pond"]) + 
                        0.03 * (water["coal-pc-sc-once-through"] + water["coal-pc-sc-pond"] + water["coal-pc-sc-tower"]),
    "Fossil Brown coal/Lignite": 0.3 * (water["coal-pc-subc-tower"] + water["coal-pc-subc-once-through"] + water["coal-pc-subc-pond"]) + 
                                 0.03 * (water["coal-pc-sc-once-through"] + water["coal-pc-sc-pond"] + water["coal-pc-sc-tower"]),
    "Fossil Coal-derived gas": water["coal-igcc-tower"],
    "Hydro Run-of-river and poundage": water["hydro"],
    "Hydro Water Reservoir": water["hydro"],
    "Solar": water["solar-pv"],
    "Wind Offshore": water["wind"],
    "Wind Onshore": water["wind"],
    "Nuclear": 0.33 * (water["nuclear-tower"] + water["nuclear-once-through"] + water["nuclear-pond"]),
    "Geothermal": 0.2 * (water["geothermal-flash-tower"] + water["geothermal-flash-dry"] + 
                          water["geothermal-binary-dry"] + water["geothermal-binary-hybrid"] + water["geothermal-egs-dry"]),
    "Waste": 0.0,
    "Fossil Oil": 0.0,
    "Hydro Pumped Storage": 0.0
}


# all the entsoe production types
production_types = [
    "Biomass",
    "Fossil Brown coal/Lignite",
    "Fossil Coal-derived gas",
    "Fossil Gas",
    "Fossil Hard coal",
    "Fossil Oil",
    "Fossil Oil shale",
    "Fossil Peat",
    "Geothermal",
    "Hydro Pumped Storage",
    "Hydro Run-of-river and poundage",
    "Hydro Water Reservoir",
    "Marine",
    "Nuclear",
    "Other renewable",
    "Solar",
    "Waste",
    "Wind Offshore",
    "Wind Onshore",
    "Other"
]

def get_generation_df(
    country: str,
    year: int
) -> pd.DataFrame:
    
    client = EntsoePandasClient(api_key=API_TOKEN)
    start_ts = pd.Timestamp(f'{year}0101', tz='Europe/Paris')
    end_ts = pd.Timestamp(f'{year+1}0101', tz='Europe/Paris')

    print(f"Requesting data from API: country = {country}; year = {year} ...")
    df = client.query_generation(country, start=start_ts, end=end_ts, psr_type=None)
    print("Data received!")

    flag = False

    # drop 'Actual Consumption' columns
    for i in df.columns:
        if i[1].strip() == "Actual Consumption":
            flag = True
            df = df.drop(i, axis = 1) 
    
    # rename columns
    if flag:
        df.columns = df.columns.map(lambda t: t[0])

    df = df.reset_index(names="start_time")

    # convert the first column to a timestamp
    df['start_time'] = pd.to_datetime(df['start_time'])

    # create end_time (define a time interval per row)
    df['end_time'] = df['start_time'].shift(-1)

    # Define the last row end_time
    last_interval = df['end_time'].iloc[-2] - df['start_time'].iloc[-2]
    df.loc[df.index[-1], 'end_time'] = df['start_time'].iloc[-1] + last_interval

    # align DataFrames with all the entsoe production types, fill missing columns with 0
    df = df.reindex(columns=['start_time','end_time']+production_types, fill_value=0)

    # convert all columns except the first/second one to numeric
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    
    return df

def make_intensities_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    df = df.copy()

    # calculate the time interval in hours
    df['duration_hours'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 3600

    # filter production types to include only those with valid carbon and water intensities
    valid_carbon_types = [ptype for ptype in carbon_intensities.keys() if ptype in df.columns]
    valid_water_types = [ptype for ptype in water_intensities.keys() if ptype in df.columns]

    # convert MW to kWh by multiplying with the duration in hours * 1000
    production_kwh = df[production_types].mul(df['duration_hours'], axis=0) * 1000

    # calculate carbon and water intensities
    df['carbon_emitted'] = production_kwh[valid_carbon_types].mul([carbon_intensities[ptype] for ptype in valid_carbon_types], axis=1).sum(axis=1)
    df['water_consumed'] = production_kwh[valid_water_types].mul([water_intensities[ptype] for ptype in valid_water_types], axis=1).sum(axis=1)

    df = df[["start_time", "end_time", "carbon_emitted", "water_consumed"]]

    return df

def divide_into_seasons(
        df: pd.DataFrame,
        year: int
) -> dict:
    df = df.copy()

    # europe/paris timezone
    bins = pd.to_datetime([
        f"{year-1}-12-31 23:59:59+01:00",
        f"{year}-03-19 23:59:59+01:00",
        f"{year}-06-19 23:59:59+01:00",
        f"{year}-09-22 23:59:59+01:00",
        f"{year}-12-20 23:59:59+01:00",
        f"{year}-12-31 23:59:59+01:00",
    ])

    labels = ["winter", "spring", "summer", "autumn", "winter"]

    df["season"] = pd.cut(df["start_time"], bins=bins, labels=labels, right=True, ordered=False)

    return {
        "summer": df[df["season"] == "summer"],
        "autumn": df[df["season"] == "autumn"],
        "winter": df[df["season"] == "winter"],
        "spring": df[df["season"] == "spring"]
    }

def fix_time_intervals(
    df: pd.DataFrame
) -> pd.DataFrame:
    # only merge consecutive rows (no splitting).
    # Accumulate rows until their total duration reaches (or exceeds)
    # the maximum interval present in the dataframe

    df = df.copy()

    df["interval_minutes"] = (df["end_time"] - df["start_time"]).dt.total_seconds() // 60
    mx = int(df["interval_minutes"].max())

    rows = []
    acc_start = None
    acc_carbon = 0.0
    acc_water = 0.0
    acc_duration = 0.0

    for _, row in df.sort_values("start_time").iterrows():
        dur = float(row["interval_minutes"])
        c = float(row.get("carbon_emitted", 0.0))
        w = float(row.get("water_consumed", 0.0))

        if acc_start is None:
            acc_start = row["start_time"]

        acc_carbon += c
        acc_water += w
        acc_duration += dur

        # when accumulated duration reaches or exceeds mx, emit a merged row
        if acc_duration >= mx:
            out_start = acc_start
            out_end = out_start + pd.Timedelta(minutes=mx)
            rows.append({
                "start_time": out_start,
                "end_time": out_end,
                "carbon_emitted": acc_carbon,
                "water_consumed": acc_water,
                "interval": mx,
            })
            # reset accumulator
            acc_start = None
            acc_carbon = 0.0
            acc_water = 0.0
            acc_duration = 0.0

    out = pd.DataFrame(rows)
    return out

def get_week(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    df = fix_time_intervals(df)

    # create helper columns
    df["year"] = df["start_time"].dt.year
    df["month"] = df["start_time"].dt.month
    df["day_of_week"] = df["start_time"].dt.dayofweek
    df["day_slot"] = df["start_time"].dt.hour * 4 + (df["start_time"].dt.minute) // 15

    week_mean = df.groupby(["day_of_week","day_slot"]).mean(numeric_only=True).reset_index()

    # rebuild a canonical Monday–Sunday timeline
    base = pd.Timestamp("2025-01-01", tz="UTC")

    def attach_canonical_week(agg_df: pd.DataFrame) -> pd.DataFrame:
        out = agg_df.copy()
        start_utc = base + pd.to_timedelta(
            out["day_of_week"] * 24 * 60 + out["day_slot"] * 15,
            unit="m"
        )
        out["start_time"] = start_utc
        out["end_time"] = start_utc + pd.to_timedelta(out["interval"], unit="m")
        return out

    week = attach_canonical_week(week_mean)

    return week[["start_time", "end_time", "carbon_emitted", "water_consumed"]]

countries = ['FR', 'DE', 'PL']
years = range(2015, 2025+1)

for country in countries:
    for year in years:
        df = get_generation_df(country, year)
        df_intensities = make_intensities_df(df)
        df_seasons = divide_into_seasons(df_intensities, year)

        for season in df_seasons.keys():
            week = get_week(df_seasons[season])
            week.to_csv(f"./generated_csv/{country}_{year}_{season}.csv", index=False)

