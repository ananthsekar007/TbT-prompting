import pandas as pd
import numpy as np

def transform(source: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    source.columns = [c.strip().lower() for c in source.columns]
    
    # Parse dates
    # Source format is like '7/3/2011'
    source['date_parsed'] = pd.to_datetime(source['date'], format='%m/%d/%Y')
    
    # Create the 'cst' column with format 'Day MM/DD/YYYY'
    # %a gives abbreviated weekday name (e.g., Sat)
    # %m gives zero-padded month
    # %d gives zero-padded day
    # %Y gives 4-digit year
    source['cst'] = source['date_parsed'].dt.strftime('%a %m/%d/%Y')
    
    # Rename value columns to target hour labels
    # value1 -> 1:00, value2 -> 2:00, ..., value24 -> 24:00
    rename_map = {}
    for i in range(1, 25):
        rename_map[f'value{i}'] = f'{i}:00'
        
    source = source.rename(columns=rename_map)
    
    # Select only the required columns
    target_cols = ['cst'] + [f'{i}:00' for i in range(1, 25)]
    
    # Ensure correct data types
    for col in target_cols[1:]:
        source[col] = pd.to_numeric(source[col], errors='coerce')
        
    # Sort by date to match target order
    source = source.sort_values(by='date_parsed')
    
    # Reset index
    source = source.reset_index(drop=True)
    
    # Return only the target columns
    return source[target_cols]
