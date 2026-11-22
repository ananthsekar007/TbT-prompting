1. Row granularity: same row count in source and target, so no aggregation; each source day becomes one target row.
2. `month` is the full month name of `DT_STRATA` right-padded with spaces to 9 characters ("January  ", "September"), i.e. strftime("%B").ljust(9).
3. `daytype` holds the single value "weekday" in every target row, so it is a constant.
4. `PCT_HOURLY_0100` ... `PCT_HOURLY_2400` map to `hr1` ... `hr24`; drop `PCT_HOURLY_2500`, `Strata_ID`, `Strata`, `DOW`, `DT_STRATA`.
5. Return exactly the target columns.
