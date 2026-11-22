import pandas as pd


def transform(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    df.columns = [c.strip() for c in df.columns]
    dates = pd.to_datetime(df["CST"].str.split().str[-1], format="%m/%d/%Y")

    out = pd.DataFrame()
    out["dt_strata"] = dates.dt.strftime("%m/%d/%Y")
    out["dow"] = dates.dt.isocalendar().day.astype(int)

    hour_cols = [c for c in df.columns if c != "CST"]
    for h, col in enumerate(hour_cols, start=1):
        out[f"pct_hourly_{h:02d}00"] = pd.to_numeric(df[col], errors="coerce")
    out["pct_hourly_2500"] = 0
    return out
