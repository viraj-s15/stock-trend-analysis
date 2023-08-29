import pandas as pd
import numpy as np
import requests
import json
from tqdm import tqdm

filename = "../preds/all_preds.csv"
df = pd.read_csv(filename)


stock_id = df["Stock ID"].to_list()


def get_trend(stock_id, array):
    for i in tqdm(range(20)):
        r = requests.get(
            f"https://frapi.marketsmojo.com/stocks_stocksid/header_info?sid={stock_id[i]}&exchange=0"
        )
        data = r.json()
        trend = data["data"]["dot_summary"]["f_txt"]
        array.append(trend)


actual_trend = []
get_trend(stock_id, actual_trend)
df2 = pd.DataFrame()
df2["Actual Trend"] = actual_trend
df.to_csv(f"../preds/actual_preds.csv")
