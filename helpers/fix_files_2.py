import os
from glob import glob
import pandas as pd

print("start clean dupes")

DATA_DIR = r"cb_data"
for path in glob(os.path.join(DATA_DIR, "*.csv")):
    df = pd.read_csv(path)
    df['ticker_time'] = pd.to_datetime(df['ticker_time'])
    df_cleaned = df.sort_values('ticker_time').groupby('candle_start', as_index=False).last()
    df_cleaned = df_cleaned[df.columns]
    print(f"{os.path.basename(path)}: original = {len(df)}, cleaned = {len(df_cleaned)}")
    df_cleaned.to_csv(path, index=False)
print("stop clean dupes")