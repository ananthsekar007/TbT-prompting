import pandas as pd


def transform(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    out = pd.DataFrame()
    dates = pd.to_datetime(df["DT_STRATA"], format="%m/%d/%Y")
    out["cst"] = dates.dt.strftime("%a %m/%d/%Y")
    for h in range(1, 25):
        out[f"{h}:00"] = pd.to_numeric(df[f"PCT_HOURLY_{h:02d}00"], errors="coerce")
    return out
