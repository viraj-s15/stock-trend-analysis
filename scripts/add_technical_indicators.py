import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Run this script after preprocessing the data, this will add the technical indicators and the output of this script
# can directly be put into the model. This script does not have any plots, for any plots refer to the notebook
# with the same name

filename = "../data/titan_weekly_data.csv"
df = pd.read_csv(filename)
df.head()


def simple_moving_average(data: pd.DataFrame, ndays: int):
    """
    Function to find the Simple Moving Average. Since this data is weekly, this variable can be updated on a weekly basis, i.e.
    the data of each week is used to compute the average
    """
    SMA = pd.Series(data["Close Price"].rolling(ndays).mean(), name="SMA")
    data = data.join(SMA)
    return data


def expo_weighted_moving_average(data, ndays):
    """
    Function to find the Simple Moving Average. Since this data is weekly, this variable can be updated on a weekly basis, i.e.
    the data of each week is used to compute the average
    """
    EMA = pd.Series(
        data["Close Price"].ewm(span=ndays, min_periods=ndays - 1).mean(),
        name="EWMA" + str(ndays),
    )
    data = data.join(EMA)
    return data


def bollinger_bands(data, window):
    """
    Function to create bollinger bands, the gap between these bands determines the volatility of that stock
    """
    MA = data["Close Price"].rolling(window).mean()
    SD = data["Close Price"].rolling(window).std()
    data["MiddleBand"] = MA
    data["UpperBand"] = MA + (2 * SD)
    data["LowerBand"] = MA - (2 * SD)
    return data


def relative_strength_index(close, periods=2):
    """
    Function to create the Relative Strength Index
    """

    close_delta = close.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


def gain(x):
    return ((x > 0) * x).sum()


def loss(x):
    return ((x < 0) * x).sum()


def mfi(high, low, close, volume, n=2):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    return (100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()


def force_index(df, ndays):
    FI = pd.Series(
        df["Close Price"].diff(ndays) * df["Total Traded Quantity"], name="ForceIndex"
    )
    df = df.join(FI)
    return df


def average_true_range(high, low, close, n=14):
    tr = np.amax(
        np.vstack(
            (
                (high - low).to_numpy(),
                (abs(high - close)).to_numpy(),
                (abs(low - close)).to_numpy(),
            )
        ).T,
        axis=1,
    )
    return pd.Series(tr).rolling(n).mean().to_numpy()


def ease_of_movement(data, ndays):
    dm = ((data["High"] + data["Low"]) / 2) - (
        (data["High"].shift(1) + data["Low"].shift(1)) / 2
    )
    br = (data["Total Traded Quantity"] / 100000000) / ((data["High"] - data["Low"]))
    EMV = dm / br
    EMV_MA = pd.Series(EMV.rolling(ndays).mean(), name="EMV")
    data = data.join(EMV_MA)
    return data


n = 5
bb = bollinger_bands(df, n)
df = bb

df["RSI"] = relative_strength_index(df["Close Price"])

df["MFI"] = mfi(
    df["High"], df["Low"], df["Close Price"], df["Total Traded Quantity"], 14
)
df = force_index(df, 1)
df["ATR"] = average_true_range(df["High"], df["Low"], df["Close Price"], 2)

df = ease_of_movement(df, 2)

df.head()

df.to_csv("../data/titan_data_indicators.csv", index=False)
