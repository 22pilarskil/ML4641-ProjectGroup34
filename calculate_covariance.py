from decimal import *
import yfinance as yf
from add_info import get_bars, writeToCSV
import csv
import pandas as pd
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
    # Add Next Day Percent Change for ND1, ND2, ND3, ND4, ND5
    df = pd.read_pickle(file)
    df = df.dropna()
    garbage = df.index[df['Close'] < 0.01].tolist()
    if len(garbage) > 0:
        df = df[:garbage[0]] # Stocks that dip below 1 penny aren't worth including in correlations
    if len(df) < 6: return None
    for i in range(5):
        df["ND" + str(i+1) + " Percent Change"] = df["Percent Change"].shift(-(i+1))
    df = df[:-5]
    X = df.to_numpy()
    N, D = X.shape
    sum_xiyi = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        temp_xiyi = np.sum(X * X[:, i, np.newaxis], axis=0)
        sum_xiyi[i] = temp_xiyi
    sum_xi = np.sum(X, axis=0)
    sum_xi2 = np.sum(np.square(X), axis=0)
    return [sum_xiyi, sum_xi, sum_xi2, N]

def get_sums_from_files(folder, files, verbose=False):
    running_sum_xiyi = None
    running_sum_xi = None
    running_sum_xi2 = None
    running_N = None
    start = "EIUWFIUENSK"
    for file in files:
        if verbose and not file.startswith(start):
            print(file)
            start = file[0]
        path = os.path.join(folder, file)
        sums = get_sums_from_pickled(path)
        if sums is None: continue
        
        if running_sum_xiyi is None:
            [running_sum_xiyi, running_sum_xi, running_sum_xi2, running_N] = sums
        else:
            running_sum_xiyi += sums[0]
            running_sum_xi += sums[1]
            running_sum_xi2 += sums[2]
            running_N += sums[3]
    
    return [running_sum_xiyi, running_sum_xi, running_sum_xi2, running_N]

def correlate_files(folder, files, verbose=False):
    [total_sum_xiyi, total_sum_xi, total_sum_xi2, total_N] = get_sums_from_files(folder, files, verbose)
    
    # https://mathoverflow.net/questions/57908/combining-correlation-coefficients
    corrs = np.zeros(total_sum_xiyi.shape, dtype=np.float64)
    D, _ = corrs.shape
    for i in range(D):
        for j in range(D):
            numerator = total_N * total_sum_xiyi[i,j] - total_sum_xi[i] * total_sum_xi[j]
            denominator = np.sqrt(total_N*total_sum_xi2[i] - np.square(total_sum_xi[i])) * np.sqrt(total_N*total_sum_xi2[j] - np.square(total_sum_xi[j]))
            corrs[i,j] = numerator/denominator
    # numerator = total_N * total_sum_xiyi - total_sum_xi * total_sum_xi[-1]
    # denominator = np.sqrt(total_N*total_sum_xi2 - np.square(total_sum_xi)) * np.sqrt(total_N*total_sum_xi2[-1] - np.square(total_sum_xi[-1]))
    # corrs = numerator/denominator
    return corrs

def get_max(folder, files):
    curr_max = np.zeros(15)
    startsWith = "X"
    
    total_stocks = 0
    total_over = 0
    for file in files:
        # if file == "TOPS.pkl": continue
        if not file.startswith(startsWith):
            print(file)
            startsWith = file[0]
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        if df['Market Cap'].isnull().values.any():
            df['Market Cap'] = 0
            df['Log Market Cap'] = 0
        df = df.dropna()
        data = df.to_numpy()
        if len(data) == 0: continue
        maxes = data.max(axis=0)
        total_stocks += 1
        if maxes[0] > 1e2:
            # print("Max close of " + '{:.2e}'.format(maxes[0]) + " on ticker: " + file.split(".")[0])
            total_over += 1
            continue
        curr_max = np.maximum(maxes, curr_max)
        
        if np.isnan(curr_max).any():
            print(df)
            print(df.columns)
            print(curr_max)
            print(file)
            return
    for name,max in zip(df.columns, curr_max):
        print(name + ": " + '{:.2e}'.format(max))
    print(df.columns)
    print(curr_max)
    print("total stocks: " + str(total_stocks))
    print("total over: " + str(total_over))

def get_count_na(folder, files):
    startsWith = "X"
    total = 0
    loss = 0
    for file in files:
        if not file.startswith(startsWith):
            print(file)
            startsWith = file[0]
        
        path = os.path.join(folder, file)
        df_a = pd.read_pickle(path)
        orig = len(df_a)
        df = df_a.dropna()
        diff = orig - len(df)
        total += orig
        loss += diff
        if diff > 0:
            print(df_a.isnull().sum())
    print(total)
    print(loss)
    print((total-loss)/total)

def add_pct_changes(folder, files, target_folder="data/NumericalData_pct"):
    pd.options.mode.chained_assignment = None
    startsWith = "X"
    for file in files:
        if not file.startswith(startsWith):
            print(file)
            startsWith = file[0]
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        df["Close pct"] = df["Close"].pct_change(fill_method=None) * 100
        pct_df = df.iloc[:,-1:]
        if len(pct_df) == 0: continue
        pct_df["Volume pct"] = df["Volume"].pct_change() * 100
        pct_df["Market Cap pct"] = df["Market Cap"].pct_change(fill_method=None) * 100
        pct_df["Log Close pct"] = df["Log Close"].pct_change(fill_method=None) * 100
        pct_df["Log Volume pct"] = df["Log Volume"].pct_change() * 100
        pct_df["Log Market Cap pct"] = df["Log Market Cap"].pct_change(fill_method=None) * 100
        pct_df["Volatility pct"] = df["Volatility"].pct_change(fill_method=None) * 100
        pct_df["RSI pct"] = df["RSI"].pct_change() * 100
        pct_df["SMA_10 pct"] = df["SMA_10"].pct_change(fill_method=None) * 100
        pct_df["SMA_20 pct"] = df["SMA_20"].pct_change(fill_method=None) * 100
        pct_df["EMA_12 pct"] = df["EMA_12"].pct_change() * 100
        pct_df["EMA_26 pct"] = df["EMA_26"].pct_change() * 100
        pct_df["MACD pct"] = df["MACD"].pct_change() * 100
        pct_df["Signal_Line pct"] = df["Signal_Line"].pct_change() * 100
        pct_df = pct_df[1:]
        pct_df.fillna(0, inplace=True)
        pd.to_pickle(pct_df, os.path.join(target_folder, file))

def process_data(folder, target_folder="data/NumericalData_processed"):
    #['Close', 'Volume', 'Market Cap', 'Log Close', 'Log Volume',
    #   'Log Market Cap', 'Volatility', 'RSI', 'Percent Change', 'SMA_10',
    #   'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line']
    # [Percent Change, Log(Volume+1), Log(Market Cap + 1), Volatility/Close, RSI, (SMA_10 - SMA_20)/SMA_20?, MACD vs Signal Line switches]
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        df = filter_pickle(df)
        if len(df) < 30:
            print(f"skipping ticker: {file.split(".")[0]}")
            continue
        move_cols_to_end = ['Percent Change', 'Log Volume', 'Log Market Cap', 'RSI']
        df = df[[c for c in df if c not in move_cols_to_end] + move_cols_to_end]
        df['Log Market Cap'] = df['Log Market Cap'].fillna(0)
        df['Volatility/Close'] = df['Volatility']/df['Close']
        df['SMA_Ratio'] = (df['SMA_10'] - df['SMA_20']) / df['SMA_20']
        df['SL_MACD'] = (df['Signal_Line'].shift(1) - df['MACD'].shift(1)) + (df['MACD'] - df['Signal_Line'])
        df = df.iloc[20:, -7:]
        
        df.to_pickle(os.path.join(target_folder, file))

def filter_pickle(df: pd.DataFrame):
    # Remove all dates before 2008 -- we don't have headlines and inflation really messes with data
    date_index = df.index[df.index > pd.Timestamp(year=2008, month=1, day=1)]
    if date_index.empty:
        return date_index
    date_index = date_index[0]
    df = df[df.index > date_index]

    # If you have nan close, there's no data for that day period. Remove it and recompute surrounding values
    indices = df.index[df["Close"].isnull()]
    df = df.drop(indices)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Now remove everything that occurs after a close drops below $0.01 -- too volatile to be reliable data
    ind = df[df['Close'] < 0.01].head(1)
    if ind.empty:
        return df
    iind = df.index.get_loc(ind.index[0])
    df = df.iloc[:iind]
    
    return df

def count_number_nans(folder, columns):
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        ticker = file.split(".")[0]
        df = pd.read_pickle(path)
        if len(columns) == 0:
            num_na = df.isnull().sum().sum()
            if num_na > 0:
                print(f"ticker {ticker}, num_nan:\n{df.isnull().sum()}")
        for col in columns:
            num_nulls = df[col].isnull().sum()
            if num_nulls > 0:
                print(f"ticker {ticker}, num_nan_{col}: {num_nulls}")
                # filter_pickle(df)

def find_unconnected_tickers():
    folder = "data/NumericalData_processed"
    files = os.listdir(folder)
    headlines = "data/cleaned_headlines.pkl"
    tickers = [a.split(".")[0] for a in files]
    df = pd.read_pickle(headlines)
    headline_tickers = df["stock"].unique()
    all = []
    for headline_ticker in headline_tickers:
        if headline_ticker not in tickers:
            all.append(headline_ticker)
    print(len(all))
    return all

def move_pkl_files():
    folder_src = "Numerical_Data"
    folder_tgt = "data/NumericalData_raw"
    files_src = os.listdir(folder_src)
    files_tgt = os.listdir(folder_tgt)
    for file in files_src:
        print("transferring " + file)
        df = pd.read_pickle(os.path.join(folder_src, file))
        pd.to_pickle(df, os.path.join(folder_tgt, file))

def find_maxes(folder): 
    total_maxes = None
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        if total_maxes is None:
            total_maxes = df.max(axis=0)
        else:
            temp_maxes = df.max(axis=0)
            # if temp_maxes.ge(6e5).any():
                # print(file)
            if temp_maxes.iloc[0] > 5e5:
                print(file)
                continue
            total_maxes = pd.concat([temp_maxes, total_maxes], axis=1).max(axis=1)
        
    
    print(total_maxes)
    return total_maxes
        
def main():
    # folder = "data/NumericalData_raw"
    # process_data(folder)
    folder = "data/NumericalData_processed"
    # find_maxes(folder)

    df = pd.read_pickle(os.path.join(folder, "ITC.pkl"))
    print(df.columns)
    # ind = df[df["Percent Change"] > 5e5].index
    # print(ind)
    # df = df.iloc[1363:1380]
    # print(df[["Close", "Percent Change", "Volume"]])


if __name__ == "__main__":
    main()
