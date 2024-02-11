from datetime import datetime, timedelta
from decimal import *
import yfinance as yf
from add_info import get_bars, writeToCSV
import csv
import pandas as pd
import os
import asyncio
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def pickle_data():
    with open("analyst_ratings_5col_fixed_headlines.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        line = next(reader)
        all_tickers = []
        startingLetter = ''
        while line is not None:
            ticker = line[3]
            if ticker in all_tickers:
                line = next(reader, None)
                continue
            all_tickers.append(ticker)
            if ticker[0] != startingLetter:
                print(ticker)
                startingLetter = ticker[0]
            line = next(reader, None)
        
        startingFromTicker = "CUT"
        all_tickers = all_tickers[all_tickers.index(startingFromTicker)+1:]
        yfTickers = yf.Tickers(" ".join(all_tickers))
        for key in yfTickers.tickers:
            ticker = key
            bars = yfTickers.tickers[key].history("max")
            if bars.empty:
                print("ticker " + ticker + ":\t Not Included")
                continue
            print("ticker " + ticker + ":\t Included")
            bars.to_pickle('pickled/'+ticker+".pkl")
            continue

def get_pickled_tickers():
    filenames = os.listdir("pickled")
    for i in range(len(filenames)):
        splits = filenames[i].split(".")
        if len(splits) > 2:
            print(filenames[i])
        filenames[i] = splits[0]
    return filenames

def add_extra_columns(df):
    # Columns are [(0): Open, (1): High, (2): Low, (3): Close, (4): Volume, (5): % Change, (6): Volatility, (7): EMA (12day), 
    #       (8): RSI, (9): MACD, (10): Signal Line, (11): SMA10, (12): SMA20, (13): Next Day % Change]
    df = df.iloc[:, :5]
    df = df.dropna()
    if len(df) < 100:
        return None
    df['Percent Change'] = df['Close'].pct_change() * 100
    df['Volatility'] = df['High'] - df['Low']
    df['EMA'] = df['Close'].ewm(span=12, adjust=False).mean()

    # Calculate RSI with initial then rolling
    delta = df['Close'].diff()
    rsi_period = 14
    gains = delta.iloc[:rsi_period].where(delta > 0, 0)
    losses = -delta.iloc[:rsi_period].where(delta < 0, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss - 0.000001 < 0: 
        initial_rsi = 100
    else:
        rs = avg_gain / avg_loss
        initial_rsi = 100 - (100 / (1 + rs))
    df.at[df.index[rsi_period-1], 'RSI'] = initial_rsi
    for i in range(rsi_period, len(df)):
        gain = delta.iloc[i] if delta.iloc[i] > 0 else 0
        loss = -delta.iloc[i] if delta.iloc[i] < 0 else 0

        avg_gain = (avg_gain * (rsi_period - 1) + gain) / rsi_period
        avg_loss = (avg_loss * (rsi_period - 1) + loss) / rsi_period

        if avg_loss - 0.000001 < 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        df.at[df.index[i], 'RSI'] = rsi
    

    EMA_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA'] - EMA_26

    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['SMA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Next Day Percent Change'] = df['Percent Change'].shift(-1)
    return df

def correlate():
    pickled_files = os.listdir("pickled")
    correlation_df = pd.read_pickle("correlations.pkl")
    # correlation_df.loc["del1"] = (np.zeros(14))
    start_at = "A"
    started = False
    for pickled_file in pickled_files:
        ticker = pickled_file.split(".")[0]
        if not started and ticker != start_at:
            continue
        started = True
        file = "pickled/"+pickled_file
        df = pd.read_pickle(file)
        if len(df) < 100:
            print("skipping " + pickled_file)
            continue
        print("correlating " + pickled_file)

        add_extra_columns(df)
        # Get rid of datapoints that have NaN or incomplete rolling windows
        X = df.iloc[27:-1, :].to_numpy()
        N, _ = X.shape
        X_bar = X - np.mean(X, axis=0)
        cov = np.matmul(X_bar.T, X_bar) / N
        cov_pchange = cov[:, -1]
        stds = np.std(X, axis=0)
        stds_combined = (stds[:, np.newaxis] * stds)[:, -1]
        # The correlations of each column with next day percent change
        corr_pchange = cov_pchange / stds_combined
        correlation_df.loc[ticker] = corr_pchange
    
    correlation_df.to_pickle("correlations.pkl")

def get_sums_from_pickled(file):
    # Columns are [(0): Open, (1): High, (2): Low, (3): Close, (4): Volume, (5): % Change, (6): Volatility, (7): EMA (12day), 
    #       (8): RSI, (9): MACD, (10): Signal Line, (11): SMA10, (12): SMA20, (13): Next Day % Change]
    df = pd.read_pickle(file)
    df = add_extra_columns(df)
    if df is None:
        return [0, 0, 0, 0]

    # Get rid of datapoints that have NaN or incomplete rolling windows
    X = df.iloc[27:-1, :].to_numpy()
    N, _ = X.shape
    sum_xiyi = np.sum(X * X[:, -1, np.newaxis], axis=0)
    sum_xi = np.sum(X, axis=0)
    sum_xi2 = np.sum(np.square(X), axis=0)
    return [sum_xiyi, sum_xi, sum_xi2, N]
    
def main():
    total_sum_xiyi = None
    total_sum_xi = None
    total_sum_xi2 = None
    total_N = None
    file_startsWith = ""
    for file in os.listdir("pickled"):
        if file_startsWith != file[0]:
            print(total_sum_xiyi)
            print(total_sum_xi)
            print(total_sum_xi2)
            print(total_N)
            print(file)
            file_startsWith = file[0]
        [sum_xiyi, sum_xi, sum_xi2, N] = get_sums_from_pickled("pickled/"+file)
        if total_sum_xiyi is None:
            total_sum_xiyi = sum_xiyi
            total_sum_xi = sum_xi
            total_sum_xi2 = sum_xi2
            total_N = N
        else:
            total_sum_xiyi += sum_xiyi
            total_sum_xi += sum_xi
            total_sum_xi2 += sum_xi2
            total_N += N
        
    # https://mathoverflow.net/questions/57908/combining-correlation-coefficients
    numerator = total_N * total_sum_xiyi - total_sum_xi * total_sum_xi[-1]
    denominator = np.sqrt(total_N*total_sum_xi2 - np.square(total_sum_xi)) * np.sqrt(total_N*total_sum_xi2[-1] - np.square(total_sum_xi[-1]))
    corrs = numerator/denominator
    print(corrs)

if __name__ == "__main__":
    main()
    # bars = yf.Ticker("TGT").history()
    # print(bars)
