import pandas as pd
import numpy as np
from tqdm import tqdm

main_df = pd.read_csv("../preds/all_preds.csv")
stock_ids = main_df["Stock ID"].to_list()
dates = main_df["Date"]
weekly_trend = [None for _ in range(len(stock_ids))]
monthly_trend = [None for _ in range(len(stock_ids))]
for i in tqdm(range(len(stock_ids))):
    df = pd.read_csv(f"../data/1000_stocks/{stock_ids[i]}.csv")
    stock_id = stock_ids[i]
    date = dates[i]
    column = df.loc[df["date"] == date]
    weekly_index_1 = len(df) - 20
    weekly_column_1 = df.loc[df.index == weekly_index_1]
    weekly_close_1 = weekly_column_1.close.to_list()[0]
    weekly_index_2 = len(df) - 19
    weekly_column_2 = df.loc[df.index == weekly_index_2]
    weekly_close_2 = weekly_column_2.close.to_list()[0]
    weekly_val = weekly_close_2 - weekly_close_1
    #######################
    monthly_index_1 = len(df) - 15
    monthly_column_1 = df.loc[df.index == monthly_index_1]
    monthly_close_1 = monthly_column_1.close.to_list()[0]
    monthly_index_2 = len(df) - 14
    monthly_column_2 = df.loc[df.index == monthly_index_2]
    monthly_close_2 = monthly_column_2.close.to_list()[0]
    monthly_val = monthly_close_2 - monthly_close_1
    if weekly_val > 0:
        weekly_trend[i] = "Up"
    elif weekly_val < 0:
        weekly_trend[i] = "Down"
    else:
        weekly_trend[i] = "Flat"

    if monthly_val > 0:
        monthly_trend[i] = "Up"
    elif monthly_val < 0:
        monthly_trend[i] = "Down"
    else:
        monthly_trend[i] = "Flat"

main_df["Actual Weekly Trend"] = weekly_trend
main_df["Actual Monthly Trend"] = monthly_trend
# print(len([i for i in monthly_trend if i is not None]))

main_df.to_csv("../preds/all_preds_with_actual.csv", index=False)
