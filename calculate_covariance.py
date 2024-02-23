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

def main():
    save_file = "data/correlations/confusion_matrix.npy"
    folder = "data/NumericalData"
    files = os.listdir(folder)
    corrs = correlate_files(folder, files, verbose=True)
    np.save(save_file, corrs)
    # x = np.load(save_file)
    # print(x)

if __name__ == "__main__":
    main()
