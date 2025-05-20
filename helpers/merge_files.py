log_file_path = "aaa.log"

print("start")

import os
from glob import glob

import pandas as pd

DATA_DIR = "..\cb_data"
files = glob(os.path.join(DATA_DIR, "*.csv"))
dfs = []
for f in files:
    df = pd.read_csv(f)
    symbol = os.path.splitext(os.path.basename(f))[0].split("_")[0]
    df["symbol"] = symbol
    dfs.append(df)

all_coins = pd.concat(dfs, ignore_index=True)
all_coins.sort_values(["symbol", "timestamp"], inplace=True)
all_coins["pct_change"] = all_coins.groupby("symbol")["price"].pct_change()

all_coins.to_csv(os.path.join(DATA_DIR, "all_cryptos_merged.csv"), index=False)

print("stop")