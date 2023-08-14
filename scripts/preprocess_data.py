import pandas as pd
import numpy as np

from NSEDownload import stocks

# ### Creating a function to streamline the process

def create_df(stock_code:str) -> pd.DataFrame:
    """
    This function is responsible for crrelianceeating the dataframe, based on the Symbol, it will create weekly data.
    """
    df = stocks.get_data(stock_symbol=stock_code, start_date='1-06-2023', end_date='7-06-2023')
    symbol = df['Symbol'].to_numpy()[0]
    high = np.max(df["High Price"].to_numpy())
    low = np.min(df["Low Price"].to_numpy())
    open_price = df['Open Price'].to_numpy()[0]
    close_price = df['Close Price'].to_numpy()[-1]
    total_quantity = np.sum(df["Total Traded Quantity"].to_numpy())
    total_val = np.sum(df["Total Traded Value"].to_numpy())
    average_trading_value = round((total_val / total_quantity),2)
    _52_week_low = df['52 Week Low Price'].to_numpy()[0] 
    _52_week_high = df['52 Week High Price'].to_numpy()[0] 
    data = [symbol,high,low,open_price,close_price,average_trading_value,total_quantity,_52_week_low,_52_week_high]
    cols = ["Symbol","High","Low","Open Price","Close Price","Average Trading Price","Total Traded Quantity","52 Week Low","52 Week High"]
    main_df = pd.DataFrame(columns=cols)
    main_df.loc[0] = data
    return main_df

def append_weekly_rows(stock_code:str,df1:pd.DataFrame,count:int,row:int,month:int,num_weeks:int,year:int) -> None:
    """normal
    Appends the weekly data to the original dataframe for upto 4 weeks,
    #########
    TODO: implement the monthly change to make it more than 4 weeks
    #########initial Commit
    """
    while num_weeks > 0:
        df2 = stocks.get_data(stock_symbol=stock_code, start_date=f'{count}-{month}-{year}', end_date=f'{count+6}-{month}-{year}')
        symbol = df2['Symbol'].to_numpy()[0]
        high = np.max(df2["High Price"].to_numpy())
        low = np.min(df2["Low Price"].to_numpy())
        open_price = df2['Open Price'].to_numpy()[0]
        close_price = df2['Close Price'].to_numpy()[-1]
        total_quantity = np.sum(df2["Total Traded Quantity"].to_numpy())
        total_val = np.sum(df2["Total Traded Value"].to_numpy())
        average_trading_value = round((total_val / total_quantity),2)
        _52_week_low = df2['52 Week Low Price'].to_numpy()[0] 
        _52_week_high = df2['52 Week High Price'].to_numpy()[0] 
        data = [symbol,high,low,open_price,close_price,average_trading_value,total_quantity,_52_week_low,_52_week_high]
        cols = ["Symbol","High","Low","Open Price","Close Price","Average Trading Price",'Total Traded Quantity',"52 Week Low","52 Week High"]
        df1.loc[row] = data
        row += 1
        count += 7
        if count >= 28:
            count = 1
            month += 1
        num_weeks -= 1

symbol = "TITAN"
# Change the symbol to and run this script to make sure the data is in the correct format
df = create_df(symbol)
# df.set_index(inplace=True)
# df.head()
len(df)

append_weekly_rows(symbol,df,1,35,1,35,2021)
# append_weekly_rows(symbol,df,1,70,1,35,2017)
# append_weekly_rows(symbol,df,1,105,1,35,2018)
# append_weekly_rows(symbol,df,1,140,1,35,2019)
# append_weekly_rows(symbol,df,1,175,1,35,2020)
# append_weekly_rows(symbol,df,1,210,1,35,2021)
# append_weekly_rows(symbol,df,1,245,1,35,2022)

len(df)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.describe()

print(f'The total entries in the data are {len(df)}')

df.to_csv("../data/titan_weekly_data.csv",index=False)



