import pandas as pd
import numpy as np


filename = "../preds/all_preds_with_actual.csv"
df = pd.read_csv(filename)


def get_weekly_acc(df=df):
    acc = 0
    pred = df['Weekly Predictions'].to_list()
    actual = df["Actual Weekly Trend"].to_list()
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            acc += 1/(len(df))
    print(f"Weekly Accuracy -> {acc}")
    return acc


def get_monthly_acc(df=df):
    acc = 0
    pred = df['Monthly Predictions']
    actual = df["Actual Monthly Trend"]
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            acc += 1/(len(df))
    print(f"Monthly Accuracy -> {acc}")
    return acc


if __name__ == "__main__":
    weekly = get_weekly_acc()
    monthly = get_monthly_acc()
    df = pd.read_csv("../preds/all_preds_with_actual.csv")
    df['Weekly Accuracy Actual'] = weekly
    df['Monthly Accuracy Actual'] = monthly
    df.to_csv("../preds/weekly_actual_with_preds.csv")