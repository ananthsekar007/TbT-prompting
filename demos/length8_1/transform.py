import pandas as pd


def transform(source: pd.DataFrame) -> pd.DataFrame:
    df = source.copy()
    ts = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M")
    df["Hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    df["value"] = pd.to_numeric(df["cerc_logger_1"], errors="coerce")

    pivot = df.pivot_table(index="Hour", columns="month", values="value", aggfunc="max")
    pivot = pivot.reindex(columns=range(1, 13))
    month_names = ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"]
    pivot.columns = month_names
    out = pivot.reset_index()
    return out[["Hour"] + month_names]
