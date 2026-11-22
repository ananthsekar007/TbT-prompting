import pandas as pd


def transform(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    dates = pd.to_datetime(df["DT_STRATA"], format="%m/%d/%Y")

    out = pd.DataFrame()
    out["month"] = dates.dt.strftime("%B").str.ljust(9)
    out["daytype"] = "weekday"
    for h in range(1, 25):
        out[f"hr{h}"] = pd.to_numeric(df[f"PCT_HOURLY_{h:02d}00"], errors="coerce")
    return out
