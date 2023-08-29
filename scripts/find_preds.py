import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import random


import random


def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


set_seeds()

# Need to reverse the db
# df = df.loc[::-1]

import os

filenames = os.listdir("../data/1000_stocks_small")


def create_preds(name: str):
    df = pd.read_csv(f"../data/1000_stocks/{name}")
    if len(df) < 30:
        return
    temp = df.copy()
    # df['close'] = (df['high'] + df['low'] )/ 2
    df.rename(
        columns={
            "open Price": "open",
            "high Price": "high",
            "low Price": "low",
            "close Price": "close",
            "Total Traded Quantity": "volume",
            "No.of Shares": "volume",
        },
        inplace=True,
    )
    # cols = ["Symbol","Ser verbose=Falseies","Prev close","Last Price","Average Price","Turnover","No. of Trades", "Deliverable Qty",'% Dly Qt to Traded Qty']
    # cols = ["WAP","No. of Trades"	,"Total Turnover (Rs.)"	,"Deliverable Quantity"	,"% Deli. Qty to Traded Qty"	,"Spread high-low"	,"Spread close-open"]

    df["EMA_9"] = df["close"].ewm(9).mean().shift()
    df["SMA_5"] = df["close"].rolling(5).mean().shift()
    df["SMA_10"] = df["close"].rolling(10).mean().shift()
    df["SMA_15"] = df["close"].rolling(15).mean().shift()
    df["SMA_30"] = df["close"].rolling(30).mean().shift()

    def relative_strength_idx(df, n=14):
        close = df["close"]
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    df["RSI"] = relative_strength_idx(df).fillna(0)

    EMA_12 = pd.Series(df["close"].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df["close"].ewm(span=26, min_periods=26).mean())
    df["MACD"] = pd.Series(EMA_12 - EMA_26)
    df["MACD_signal"] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    df["close"] = df["close"].shift(-1)

    df = df.iloc[33:]  # Because of moving average
    df = df[:-1]  # Because of shifting close price

    df.index = range(len(df))

    df.head()

    drop_cols = ["date", "volume", "open", "low", "high"]
    df.drop(columns=drop_cols, inplace=True)
    df.head()

    X = df.iloc[:, 1:]
    y = df.close

    x_len = int(0.8 * len(X))
    y_len = int(0.8 * len(y))
    X_trainval = X[:x_len]
    X_test = X[x_len:]
    y_trainval = y[:y_len]
    y_test = y[y_len:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    class RegressionDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__(self):
            return len(self.X_data)

    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    val_dataset = RegressionDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    )
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    )

    EPOCHS = 2500
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_FEATURES = len(X.columns)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    class MultipleRegression(nn.Module):
        def __init__(self, num_features):
            super(MultipleRegression, self).__init__()

            self.layer_1 = nn.Linear(num_features, 16)
            self.layer_2 = nn.Linear(16, 32)
            self.layer_3 = nn.Linear(32, 16)
            self.layer_out = nn.Linear(16, 1)

            self.relu = nn.ReLU()

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            x = self.layer_out(x)
            return x

        def predict(self, test_inputs):
            x = self.relu(self.layer_1(test_inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            x = self.layer_out(x)
            return x

    # torch.cuda.set_device("cuda:0")
    # print(torch.cuda.get_device_name())

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = MultipleRegression(NUM_FEATURES)
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_stats = {"train": [], "val": []}

    for e in tqdm(range(1, EPOCHS + 1)):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(
                device
            )
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

                val_epoch_loss += val_loss.item()
            # print(torch.cuda.get_device_name())

            loss_stats["train"].append(train_epoch_loss / len(train_loader))
            loss_stats["val"].append(val_epoch_loss / len(val_loader))

        if e % 50 == 0:
            print(
                f"Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}"
            )
    # torch.cuda.set_device("cuda:0")
    # print(torch.cuda.get_device_name())

    print(len(loss_stats["train"]))
    print(len(loss_stats["val"]))

    train_val_loss_df = (
        pd.DataFrame.from_dict(loss_stats)
        .reset_index()
        .melt(id_vars=["index"])
        .rename(columns={"index": "epochs"})
    )
    plt.figure(figsize=(15, 8))
    sns.lineplot(
        data=train_val_loss_df, x="epochs", y="value", hue="variable"
    ).set_title("Train-Val Loss/Epoch")

    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list.append(y_test_pred.cpu().numpy())
            y_pred_list = [a for a in y_pred_list]

    y_pred = []
    for i in tqdm(range(len(y_pred_list))):
        for j in range(len(y_pred_list[i])):
            for k in range(len(y_pred_list[i][j])):
                y_pred.append(y_pred_list[i][j][k])

    print(y_pred)
    print(y_test.tolist())
    y_pred_np = np.array(y_pred)
    difference = np.subtract(y_pred_np, y_test)
    difference = abs(difference)
    mean_difference = np.mean(difference)
    print(f"The average deviation in error is: {mean_difference}")
    sns.lineplot(difference).set(title="Deviation of Error")

    mse = mean_squared_error(y_test, y_pred)
    r_square = r2_score(y_test, y_pred)
    mean_abs_err = mean_absolute_percentage_error(y_test, y_pred)
    print("Mean Squared Error :", mse)
    print("R^2 :", r_square)
    print(f"Accuracy (using MSE): {(100 - mse)}%")
    print(f"Mean absolute percentage error: {100 - mean_abs_err}%")

    temp_len = len(y_pred)
    final_len = len(temp) - temp_len
    temp = temp[final_len:].copy()
    temp["close"] = y_test
    close = temp.close.to_list()
    weekly_change = []
    for i in range(len(close) - 7):
        weekly_change.append(close[i] - close[i + 7])
    for i in range(len(close) - 7, len(close)):
        weekly_change.append(weekly_change[i - 7])

    weekly_trend = []
    for i in weekly_change:
        if i > 0:
            weekly_trend.append("Up")
        elif i < 0:
            weekly_trend.append("Down")
        else:
            weekly_trend.append("Flat")
    temp["Weekly Trend"] = weekly_trend

    monthly_change = []
    for i in range(len(close) - 30):
        monthly_change.append(close[i] - close[i + 30])
    for i in range(len(close) - 30, len(close)):
        monthly_change.append(weekly_change[i - 30])

    monthly_trend = []
    for i in monthly_change:
        if i > 0:
            monthly_trend.append("Up")
        elif i < 0:
            monthly_trend.append("Down")
        else:
            monthly_trend.append("Flat")
    temp["Monthly Trend"] = monthly_trend

    preds = y_pred
    weekly_preds_change = []
    for i in range(len(preds) - 7):
        weekly_preds_change.append(preds[i] - preds[i + 7])
    for i in range(len(preds) - 7, len(preds)):
        weekly_preds_change.append(weekly_preds_change[i - 7])

    weekly_preds_trend = []
    for i in weekly_preds_change:
        if i > 0:
            weekly_preds_trend.append("Up")
        elif i < 0:
            weekly_preds_trend.append("Down")
        else:
            weekly_preds_trend.append("Flat")
    temp["Weekly Predictions Trend"] = weekly_preds_trend

    monthly_preds_change = []
    for i in range(len(preds) - 30):
        monthly_preds_change.append(preds[i] - preds[i + 30])
    for i in range(len(preds) - 30, len(preds)):
        monthly_preds_change.append(monthly_preds_change[i - 30])

    monthly_preds_trend = []
    for i in monthly_preds_change:
        if i > 0:
            monthly_preds_trend.append("Up")
        elif i < 0:
            monthly_preds_trend.append("Down")
        else:
            monthly_preds_trend.append("Flat")
    temp["Monthly Preds Trend"] = monthly_preds_trend

    weekly_accuracy = 0
    weekly = temp["Weekly Trend"].to_list()
    weekly_preds = temp["Weekly Predictions Trend"].to_list()
    for i in range(len(weekly)):
        if weekly[i] == weekly_preds[i]:
            weekly_accuracy += 1 / len(weekly)
    print(weekly_accuracy)

    monthly_accuracy = 0
    monthly = temp["Monthly Trend"].to_list()
    monthly_preds = temp["Monthly Preds Trend"].to_list()
    for i in range(len(monthly)):
        if monthly[i] == monthly_preds[i]:
            monthly_accuracy += 1 / len(weekly)
    print(monthly_accuracy)

    temp["Weekly Accuracy"] = weekly_accuracy
    temp["Monthly Accuracy"] = monthly_accuracy

    temp["Total Acc"] = (weekly_accuracy + monthly_accuracy) / 2

    temp.to_csv(f"../preds/1000_companies_preds/{name}", index=False)


for i in filenames:
    # df = pd.read_csv(f"../data/1000_stocks/{i}")
    # if len(df) < 350: print(i)
    # if len(df) < 350: os.remove(f"../data/1000_stocks/{i}")
    # if len(df) < 350: print(i)
    create_preds(i)
