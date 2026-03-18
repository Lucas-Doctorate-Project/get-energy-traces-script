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
with open('intensities.json', 'r') as f:
    data = json.load(f)

# extract carbon and water intensities from the JSON file
carbon = data['carbon']
water = data['water']

# define dictionaries for carbon and water intensities
# https://colab.research.google.com/drive/1vPR_nndzlkHKROMDinlpXU3XWIxTfFQx?usp=sharing


carbon_intensities = {
    "Biomass": carbon["biomass-ipcc-2014"],
    "Fossil Gas": carbon["gas-ngcc-unece-2020"],
    "Fossil Hard coal": 0.95 * carbon["coal-pc-unece-2020"] + 0.05 * carbon["coal-sc-unece-2020"],
    "Fossil Brown coal/Lignite": 0.95 * carbon["coal-pc-unece-2020"] + 0.05 * carbon["coal-sc-unece-2020"],
    "Fossil Coal-derived gas": carbon["coal-igcc-unece-2020"],
    "Hydro Run-of-river and poundage": 0.95 * carbon["hydro-medium-unece-2020"] + 0.05 * carbon["hydro-large-unece-2020"],
    "Hydro Water Reservoir": 0.95 * carbon["hydro-medium-unece-2020"] + 0.05 * carbon["hydro-large-unece-2020"],
    "Solar": (
        0.45 * carbon["solar-pv-poly-si-roof-unece-2020"]
        + 0.45 * carbon["solar-pv-poly-si-ground-unece-2020"]
        + 0.025
        * (
            carbon["solar-pv-cdte-ground-unece-2020"]
            + carbon["solar-pv-cdte-roof-unece-2020"]
            + carbon["solar-pv-cigs-ground-unece-2020"]
            + carbon["solar-pv-poly-si-roof-unece-2020"]
        )
    ),
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
    "Fossil Hard coal": (
        0.3
        * (
            water["coal-pc-subc-tower"]
            + water["coal-pc-subc-once-through"]
            + water["coal-pc-subc-pond"]
        )
        + 0.03
        * (
            water["coal-pc-sc-once-through"]
            + water["coal-pc-sc-pond"]
            + water["coal-pc-sc-tower"]
        )
    ),
    "Fossil Brown coal/Lignite": (
        0.3
        * (
            water["coal-pc-subc-tower"]
            + water["coal-pc-subc-once-through"]
            + water["coal-pc-subc-pond"]
        )
        + 0.03
        * (
            water["coal-pc-sc-once-through"]
            + water["coal-pc-sc-pond"]
            + water["coal-pc-sc-tower"]
        )
    ),
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

    total = df[production_types].sum(axis=1)
    # avoid division by zero: use NaN for division then fill with 0
    total_safe = total.replace(0, np.nan)
    shares = df[production_types].div(total_safe, axis=0).fillna(0)
    df[production_types] = shares

    # filter production types to include only those with valid carbon and water intensities
    valid_carbon_types = [ptype for ptype in carbon_intensities.keys() if ptype in df.columns]
    valid_water_types = [ptype for ptype in water_intensities.keys() if ptype in df.columns]

    # calculate carbon and water intensities
    df['carbon_intensity'] = df[valid_carbon_types].mul([carbon_intensities[ptype] for ptype in valid_carbon_types], axis=1).sum(axis=1)
    df['water_intensity'] = df[valid_water_types].mul([water_intensities[ptype] for ptype in valid_water_types], axis=1).sum(axis=1)

    df = df[["start_time", "end_time", "carbon_intensity", "water_intensity"]]

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
) -> tuple[pd.DataFrame, int]:
    df = df.copy()

    # Normalize to UTC before computing intervals/slots (avoids DST distortions)
    def _to_utc(ts: pd.Series) -> pd.Series:
        out = pd.to_datetime(ts)
        if out.dt.tz is None:
            out = out.dt.tz_localize(
                "Europe/Paris",
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        return out.dt.tz_convert("UTC")

    df["start_time"] = _to_utc(df["start_time"])
    df["end_time"] = _to_utc(df["end_time"])

    target_minutes = 60

    interval_series = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    df["interval_minutes"] = interval_series.round().astype(int)
    if (df["interval_minutes"] <= 0).any():
        raise ValueError("Found non-positive time interval in data")

    df = df.sort_values("start_time").reset_index(drop=True)


    rows: list[dict] = []
    i = 0
    n = len(df)

    while i < n:
        row = df.iloc[i]
        dur = int(row["interval_minutes"])
        c = float(row.get("carbon_intensity", 0.0))
        w = float(row.get("water_intensity", 0.0))

        if dur == target_minutes:
            rows.append({
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "carbon_intensity": c,
                "water_intensity": w,
                "interval": target_minutes,
            })
            i += 1
            continue

        if dur < target_minutes:
            if target_minutes % dur != 0:
                raise ValueError(
                    f"Interval {dur} minutes does not divide {target_minutes}; cannot normalize to {target_minutes} minutes"
                )
            group_size = target_minutes // dur

            # Fast path: strict aggregation of exactly group_size equal-duration rows.
            if i + group_size <= n:
                grp = df.iloc[i : i + group_size]
                if (grp["interval_minutes"] == dur).all():
                    starts = grp["start_time"].to_list()
                    ends = grp["end_time"].to_list()
                    for j in range(1, len(grp)):
                        if starts[j] != ends[j - 1]:
                            raise ValueError(
                                "Non-contiguous time intervals while aggregating to 60 minutes; cannot safely average"
                            )

                    rows.append({
                        "start_time": grp.iloc[0]["start_time"],
                        "end_time": grp.iloc[-1]["end_time"],
                        "carbon_intensity": float(grp["carbon_intensity"].mean()),
                        "water_intensity": float(grp["water_intensity"].mean()),
                        "interval": target_minutes,
                    })
                    i += group_size
                    continue

            # Fallback: accumulate contiguous rows until total duration hits a multiple
            # of 60 minutes; then take a duration-weighted mean and replicate per hour.
            block_start = df.iloc[i]["start_time"]
            current_end = df.iloc[i]["start_time"]
            total_span = 0
            block_end_index = None

            for j in range(i, n):
                r = df.iloc[j]
                if j > i and r["start_time"] != current_end:
                    raise ValueError(
                        "Non-contiguous time intervals while normalizing to 60 minutes; cannot safely average"
                    )
                d = int(r["interval_minutes"])
                if d <= 0:
                    raise ValueError("Found non-positive time interval in data")

                total_span += d
                current_end = r["end_time"]

                if total_span >= target_minutes and (total_span % target_minutes) == 0:
                    block_end_index = j
                    break

            if block_end_index is None:
                raise ValueError(
                    f"Could not accumulate a contiguous block to a multiple of {target_minutes} minutes starting at {block_start}"
                )

            block = df.iloc[i : block_end_index + 1]
            block_start_ts = block.iloc[0]["start_time"]
            block_end_ts = block.iloc[-1]["end_time"]
            print(
                "[fix_time_intervals] Fallback used: merged "
                f"{len(block)} rows (from {block_start_ts} to {block_end_ts}) into "
                f"{total_span // target_minutes} hour(s) before splitting to 60-minute rows"
            )
            for _, r in block.iterrows():
                print(
                    "[fix_time_intervals]   merged-row: "
                    f"start_time={r['start_time']}, end_time={r['end_time']}, interval={int(r['interval_minutes'])}min"
                )
            weights = block["interval_minutes"].astype(float)
            carbon_avg = float((block["carbon_intensity"] * weights).sum() / weights.sum())
            water_avg = float((block["water_intensity"] * weights).sum() / weights.sum())

            repeat = total_span // target_minutes
            for k in range(repeat):
                out_start = block_start + pd.Timedelta(minutes=target_minutes * k)
                out_end = out_start + pd.Timedelta(minutes=target_minutes)
                rows.append({
                    "start_time": out_start,
                    "end_time": out_end,
                    "carbon_intensity": carbon_avg,
                    "water_intensity": water_avg,
                    "interval": target_minutes,
                })

            i = block_end_index + 1
            continue

        # dur > target_minutes
        if dur % target_minutes != 0:
            raise ValueError(
                f"Interval {dur} minutes is not a multiple of {target_minutes}; cannot split into {target_minutes}-minute steps"
            )

        repeat = dur // target_minutes
        base_start = row["start_time"]
        for k in range(repeat):
            out_start = base_start + pd.Timedelta(minutes=target_minutes * k)
            out_end = out_start + pd.Timedelta(minutes=target_minutes)
            rows.append({
                "start_time": out_start,
                "end_time": out_end,
                "carbon_intensity": c,
                "water_intensity": w,
                "interval": target_minutes,
            })
        i += 1

    out = pd.DataFrame(rows)
    return out, target_minutes

def get_week(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    df, mx = fix_time_intervals(df)

    # create helper columns
    df["year"] = df["start_time"].dt.year
    df["month"] = df["start_time"].dt.month
    df["day_of_week"] = df["start_time"].dt.dayofweek
    # slot index within the day, based on the normalized interval (mx minutes)
    slots_per_hour = 60 // mx
    df["day_slot"] = df["start_time"].dt.hour * slots_per_hour + (df["start_time"].dt.minute) // mx

    week_mean = df.groupby(["day_of_week","day_slot"]).mean(numeric_only=True).reset_index()

    # rebuild a canonical Monday–Sunday timeline
    base = pd.Timestamp("2025-01-01", tz="UTC")

    def attach_canonical_week(agg_df: pd.DataFrame) -> pd.DataFrame:
        out = agg_df.copy()
        start_utc = base + pd.to_timedelta(
            out["day_of_week"] * 24 * 60 + out["day_slot"] * mx,
            unit="m"
        )
        out["start_time"] = start_utc
        out["end_time"] = start_utc + pd.to_timedelta(out["interval"], unit="m")
        return out

    week = attach_canonical_week(week_mean)

    return week[["start_time", "end_time", "carbon_intensity", "water_intensity"]], mx


def format_for_seconds_export(
    df: pd.DataFrame,
    interval_minutes: int,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out.insert(
        0,
        "timestamp",
        (np.arange(len(out), dtype=np.int64) * int(interval_minutes) * 60).astype(np.int64),
    )
    # Keep only one time column (timestamp), plus intensities
    cols = ["timestamp"]
    for c in ["carbon_intensity", "water_intensity"]:
        if c in out.columns:
            cols.append(c)
    return out[cols]

countries = ['PL', 'DE']
years = range(2015, 2025+1)

for country in countries:
    season_weeks: dict[str, list[pd.DataFrame]] = {"summer": [], "autumn": [], "winter": [], "spring": []}
    season_mx: dict[str, int] = {}

    for year in years:
        df = get_generation_df(country, year)

        df_intensities = make_intensities_df(df)
        df_seasons = divide_into_seasons(df_intensities, year)

        for season in df_seasons.keys():
            # fixed to 60-minute resolution, then build canonical week
            week, mx = get_week(df_seasons[season])
            # week.to_csv(f"./generated_csv/{country}_{year}_{season}_res={mx}min.csv", index=False)

            season_weeks[season].append(week)
            if season in season_mx and season_mx[season] != mx:
                raise ValueError(
                    f"Inconsistent normalized interval for {country} {season}: {season_mx[season]} vs {mx}"
                )
            season_mx[season] = mx

    # Merge across years: for each equivalent hour (canonical start_time), take mean across years
    min_year = min(years)
    max_year = max(years)
    for season, weeks_list in season_weeks.items():
        if not weeks_list:
            continue

        mx = season_mx.get(season, 60)
        merged = pd.concat(weeks_list, ignore_index=True)
        merged_mean = (
            merged
            .groupby(["start_time"], as_index=False)
            .mean(numeric_only=True)
            .sort_values("start_time")
        )
        merged_mean["end_time"] = merged_mean["start_time"] + pd.to_timedelta(mx, unit="m")
        merged_mean = merged_mean[["start_time", "end_time", "carbon_intensity", "water_intensity"]]
        merged_mean = format_for_seconds_export(merged_mean, mx)
        merged_mean.to_csv(
            f"./generated_csv/{country}_{season}.csv",
            index=False,
        )

