import pandas as pd

def convert_mtu_to_utc_iana(
    df: pd.DataFrame,
    interval_col: str = "MTU (CET/CEST)",
    local_tz: str = "Europe/Paris",
) -> pd.DataFrame:
    """Parse MTU intervals and produce UTC start/end timestamps (DST-aware)."""

    out = df.copy()

    # Split "start - end" into two strings
    parts = out[interval_col].str.split(" - ", expand=True)
    start_raw = parts[0].str.strip()
    end_raw = parts[1].str.strip()

    # Drop optional "(CET)/(CEST)" labels from the strings
    start_txt = start_raw.str.replace(r"\s*\((CET|CEST)\)", "", regex=True).str.strip()
    end_txt = end_raw.str.replace(r"\s*\((CET|CEST)\)", "", regex=True).str.strip()

    # Parse dd/mm/yyyy HH:MM:SS into naive datetimes 
    start_naive = pd.to_datetime(start_txt, dayfirst=True)
    end_naive = pd.to_datetime(end_txt, dayfirst=True)

    # Attach IANA timezone rules (handles DST transitions)
    start_local = start_naive.dt.tz_localize(local_tz, ambiguous="infer", nonexistent="raise")
    end_local = end_naive.dt.tz_localize(local_tz, ambiguous="infer", nonexistent="raise")

    # Convert to UTC
    out["start_utc"] = start_local.dt.tz_convert("UTC")
    out["end_utc"] = end_local.dt.tz_convert("UTC")

    return out

def convert_to_15min_delta(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    dur = df["end_utc"] - df["start_utc"]
    mask_hour = dur == pd.Timedelta(hours=1)

    # split exactly 1h rows into four 15 min slices
    df_hour = df[mask_hour].copy()
    df_hour["start_15_list"] = df_hour["start_utc"].apply(
        lambda t: pd.date_range(t, periods=4, freq="15min")
    )
    df_hour = df_hour.explode("start_15_list").rename(columns={"start_15_list": "start_utc_15"})
    df_hour["end_utc_15"] = df_hour["start_utc_15"] + pd.Timedelta(minutes=15)
    df_hour["Generation (MW)"] = df_hour["Generation (MW)"] / 4

    # keep non-1h rows as-is (assumed already 15 min)
    df_other = df[~mask_hour].copy()
    df_other = df_other.rename(columns={"start_utc": "start_utc_15", "end_utc": "end_utc_15"})

    df15_final = pd.concat(
        [df_hour, df_other],
        ignore_index=True
    )[["start_utc_15", "end_utc_15", "Area", "Production Type", "Generation (MW)"]]

    assert (
        (df15_final["end_utc_15"] - df15_final["start_utc_15"]) == pd.Timedelta(minutes=15)
    ).all(), "convert_to_15min_delta: non-15min interval detected"

    return df15_final
    
def creates_15min_delta_pivoted_table(
    df: pd.DataFrame,
    local_tz: str,
) -> pd.DataFrame:
    df = df.copy()

    # convert genaration (MW) to numeric numbers.
    # if the to-be-converted value is a non-numeric one, put 0 
    df["Generation (MW)"] = pd.to_numeric(df["Generation (MW)"], errors="coerce")
    df = df.fillna(0)

    # convert to UTC
    df = convert_mtu_to_utc_iana(df, local_tz=local_tz)

    # convert from 1h time delta to 15min
    df = convert_to_15min_delta(df)

    # pivot it so each prod type has its own column
    df = df.pivot_table(
        index=["start_utc_15", "end_utc_15", "Area"],
        columns="Production Type",
        values="Generation (MW)"
    )
    df = df.reset_index()

    return df



summer_months = [6, 7, 8] # june, july, august
winter_months = [12, 1, 2] # december, january, february

# the three countries uses CET timezone 
TZ_BY_COUNTRY = {
    "france": "Europe/Paris",
    "germany": "Europe/Berlin",
    "poland": "Europe/Warsaw",
}

countries = ['france', 'poland', 'germany']
years = ['2023', '2024', '2025']

for country in countries:
    print(f"Processing: {country}")

    frames = []

    local_tz = TZ_BY_COUNTRY.get(country.lower(), "UTC")

    for year in years:
        filename = f'./raw_data/{country}{year}.csv'
        year_df = pd.read_csv(filename, low_memory=False)
        year_df = creates_15min_delta_pivoted_table(year_df, local_tz=local_tz)
        frames.append(year_df)

        print(f"{filename} processed.")

    if not frames:
        print(f"No data found for {country}, skipping.")
        continue

    df = pd.concat(frames, ignore_index=True)

    # create helper columns
    df["year"] = df["start_utc_15"].dt.year
    df["month"] = df["start_utc_15"].dt.month
    df["day_of_week"] = df["start_utc_15"].dt.dayofweek
    df["day_slot"] = df["start_utc_15"].dt.hour * 4 + (df["start_utc_15"].dt.minute) // 15

    # split into seasonal subsets
    summer = df[df['month'].isin(summer_months)].copy()
    winter = df[df['month'].isin(winter_months)].copy()

    # groupby and calculate mean for the production values
    summer_mean = summer.groupby(["Area","day_of_week","day_slot"]).mean(numeric_only=True).reset_index()
    winter_mean = winter.groupby(["Area","day_of_week","day_slot"]).mean(numeric_only=True).reset_index()

    # rebuild a canonical Monday–Sunday timeline
    base = pd.Timestamp("2025-01-01", tz="UTC")

    def attach_canonical_week(agg_df: pd.DataFrame) -> pd.DataFrame:
        out = agg_df.copy()
        start_utc_15 = base + pd.to_timedelta(
            out["day_of_week"] * 24 * 60 + out["day_slot"] * 15,
            unit="m"
        )
        out["start_utc_15"] = start_utc_15
        out["end_utc_15"] = start_utc_15 + pd.Timedelta(minutes=15)
        return out

    summer_week = attach_canonical_week(summer_mean)
    winter_week = attach_canonical_week(winter_mean)

    # drop helper columns
    helper_cols = ["year", "month", "day_of_week", "day_slot"]
    summer_week = summer_week.drop(columns=[c for c in helper_cols if c in summer_week.columns])
    winter_week = winter_week.drop(columns=[c for c in helper_cols if c in winter_week.columns])

    # round production values to 2 decimals
    summer_value_cols = summer_week.select_dtypes(include="float").columns
    winter_value_cols = winter_week.select_dtypes(include="float").columns
    summer_week[summer_value_cols] = summer_week[summer_value_cols].round(2)
    winter_week[winter_value_cols] = winter_week[winter_value_cols].round(2)

    # sort by start_utc, ensure time columns come first
    summer_week = summer_week.sort_values(["start_utc_15"])
    winter_week = winter_week.sort_values(["start_utc_15"])

    def reorder(df):
        lead = ["start_utc_15", "end_utc_15"]
        rest = [c for c in df.columns if c not in lead]
        return df[lead + rest]

    summer_week = reorder(summer_week)
    winter_week = reorder(winter_week)

    # save
    summer_week.to_csv(f'./generated_csv/{country}_summer.csv', index=False)
    winter_week.to_csv(f'./generated_csv/{country}_winter.csv', index=False)

    print(20*"--/")




