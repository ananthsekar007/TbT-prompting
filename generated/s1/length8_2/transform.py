import pandas as pd
import numpy as np

def transform(source: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    source.columns = [c.strip().lower() for c in source.columns]
    
    # Identify the value column (assuming it's the one that isn't 'date')
    value_col = [c for c in source.columns if c != 'date'][0]
    
    # Parse dates
    df = source.copy()
    df['datetime'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M', errors='coerce')
    
    # Drop rows with invalid dates or null values
    df = df.dropna(subset=['datetime', value_col])
    
    # Extract hour and month
    df['Hour'] = df['datetime'].dt.hour
    df['Month'] = df['datetime'].dt.month
    
    # Group by month and hour, take mean
    grouped = df.groupby(['Month', 'Hour'])[value_col].mean()
    
    # Pivot to have months as columns
    pivoted = grouped.unstack(level='Month')
    
    # Reindex to ensure all 24 hours are present
    pivoted = pivoted.reindex(range(24))
    
    # Rename month columns to full names
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    pivoted.columns = [month_names.get(c, c) for c in pivoted.columns]
    
    # Reset index to make Hour a column
    out = pivoted.reset_index()
    out.columns = ['Hour'] + list(out.columns[1:])
    
    # Ensure correct column order
    target_cols = ['Hour', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    # Add any missing month columns as NaN
    for col in target_cols:
        if col not in out.columns:
            out[col] = np.nan
            
    return out[target_cols]
