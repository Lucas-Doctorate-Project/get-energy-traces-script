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

    # defines 15 min intervals and 'explodes it' (creates 4 new lines)
    df["start_15_list"] = df["start_utc"].apply(lambda t: pd.date_range(t, periods=4, freq="15min"))
    df15 = df.explode("start_15_list").rename(columns={"start_15_list": "start_utc_15"})

    # defines new end_time = start_time + 15min
    df15["end_utc_15"] = df15["start_utc_15"] + pd.Timedelta(minutes=15)

    # divides generations values by 4
    df15["Generation (MW)"] = df15["Generation (MW)"] / 4

    # select the desired columns
    df15_final = df15[["start_utc_15", "end_utc_15", "Area", "Production Type", "Generation (MW)"]]
    
    return df15_final
    
def creates_15min_delta_pivoted_table(
    df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    # convert genaration (MW) to numeric numbers.
    # if the to-be-converted value is a non-numeric one, put 0 
    df["Generation (MW)"] = pd.to_numeric(df["Generation (MW)"], errors="coerce")
    df = df.fillna(0)

    # convert to UTC
    df = convert_mtu_to_utc_iana(df)

    # check if all the intervals are 1h
    dur = df["end_utc"] - df["start_utc"]
    assert (dur == pd.Timedelta(hours=1)).all(), (
        f"Found non-1h intervals. Counts:\n{dur.value_counts().head(10)}"
    )

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

# read csv
df = pd.read_csv('france2023.csv')

df = creates_15min_delta_pivoted_table(df)

df.to_csv('france2023_15min.csv', index=False)



