1. Row granularity: one source row per (strata, day) and the target has the same number of rows, so this is a row-preserving reshape. No grouping.
2. Key column: target `cst` looks like "Fri 01/01/2016". It comes from `DT_STRATA` ("1/1/2016"), parsed as %m/%d/%Y and re-formatted as "%a %m/%d/%Y" (abbreviated weekday, zero-padded month/day).
3. Value columns: `PCT_HOURLY_0100` ... `PCT_HOURLY_2400` map to `1:00` ... `24:00`. `PCT_HOURLY_2500` has no target counterpart and is dropped.
4. Drop `Strata_ID`, `Strata`, `DOW`; they do not appear in the target.
5. Return exactly the target columns in target order.
