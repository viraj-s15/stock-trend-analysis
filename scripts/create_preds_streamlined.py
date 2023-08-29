import os
from os import path
import shutil
import pandas as pd
import numpy as np

Source_Path = "/home/veer/Documents/mojo/stock-trend/preds/1000_companies_preds"


stock_ids = []
weekly_preds = []
monthly_preds = []
dates = []
weekly_conf = []
monthly_conf = []
for count, filename in enumerate(os.listdir(Source_Path)):
    df = pd.read_csv(f"../preds/1000_companies_preds/{filename}")
    df.sort_values(by="date", inplace=True)
    date = df.iloc[-20].date
    dot_index = filename.find(".")
    stock_id = filename[:dot_index]
    weekly_pred = df.iloc[-20]["Weekly Predictions Trend"]
    monthly_pred = df.iloc[-20]["Monthly Preds Trend"]
    monthly_acc = df.iloc[-20]["Monthly Accuracy"]
    weekly_acc = df.iloc[-20]["Weekly Accuracy"]
    stock_ids.append(stock_id)
    weekly_preds.append(weekly_pred)
    monthly_preds.append(monthly_pred)
    dates.append(date)
    monthly_conf.append(round(monthly_acc, 2))
    weekly_conf.append(round(weekly_acc, 2))

preds = pd.DataFrame()
preds["Date"] = dates
preds["Stock ID"] = stock_ids
preds["Weekly Predictions"] = weekly_preds
preds["Monthly Predictions"] = monthly_preds
preds["Weekly Confidence"] = weekly_conf
preds["Monthly Confidence"] = monthly_conf
preds.to_csv("../preds/all_preds.csv", index=False)


#
