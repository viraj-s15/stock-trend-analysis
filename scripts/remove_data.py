import pandas as pd
import numpy as np
import os
from tqdm import tqdm


filenames = os.listdir("../data/1000_stocks")
print(filenames)

for file in tqdm(filenames):
    df = pd.read_csv(f"../data/1000_stocks/{file}")
    df_len = int(0.95*len(df))
    df.sort_values(by=["date"],inplace=True)
    df = df.iloc[:df_len,:]
    df.to_csv(f"../data/1000_stocks_small/{file}", index=False)
