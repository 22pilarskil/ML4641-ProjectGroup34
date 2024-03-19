import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # Ignore deprecation warning for parsing timezone -> unix; expected behavior, they don't plan to deprecate
# branch bert_test file datasets.py
# TODO remake this into a dataset that calls a dataloader
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import chardet

class HeadlineDataset(Dataset):
    def __init__(self, headlines_file, numerical_folder, trading_days_before: int = 0, trading_days_after: int = 1):
        self.numerical_folder = numerical_folder
        self.headlines_file = headlines_file
        self.trading_days_before = trading_days_before
        self.trading_days_after = trading_days_after
        self.data = pd.read_pickle(headlines_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headline_info = self.data.iloc[idx]
        ticker = headline_info['stock']

        numerical_df = pd.read_pickle(os.path.join(self.numerical_folder, ticker + ".pkl"))

        # Problem: Ticker headline dates don't necessarily line up with trading dates
        dates = numerical_df.index.to_numpy()
        dates_unix = dates.astype('datetime64[s]').astype('int')
        headline_date_unix = np.array(headline_info.loc["date"]).astype('datetime64[s]').astype('int')
        index = np.searchsorted(dates_unix, headline_date_unix, side='right')
        slice_size = self.trading_days_after + self.trading_days_before + 1
        if len(numerical_df) <= slice_size:
            print("Fatal: Cannot create slice of length " + str(slice_size) + " for ticker " + ticker + " with numerical_data of length " + str(len(numerical_df)))
            raise Exception()
        elif index - self.trading_days_before < 0 or index + self.trading_days_after >= len(dates):
            print("Slicing issue for headline date " + str(headline_info.loc["date"]) + " in ticker " + ticker + "...approximating...")
            if index - self.trading_days_before < 0:
                index = self.trading_days_before
            else:
                index = len(dates) - self.trading_days_after
        df_slice = slice(index-self.trading_days_before, index+self.trading_days_after)
        numerical_tensor = torch.tensor(numerical_df.iloc[df_slice].values)

        return_obj = {
            'title': headline_info['title'],
            'date': headline_info['date'],
            'stock': headline_info['stock'],
            'numerical': numerical_tensor,
            'labels': torch.tensor(headline_info['percent change'], dtype=torch.float)
        }
        
        return return_obj


def create_data_loaders(headlines_file, numerical_folder, trading_days_before=0, trading_days_after=1, batch_size=32):
    # Initialize dataset
    dataset = HeadlineDataset(headlines_file=headlines_file, numerical_folder=numerical_folder, trading_days_before=trading_days_before, trading_days_after=trading_days_after)
    
    # Calculate split sizes
    train_size = int(0.8 * len(dataset))
    val_size = test_size = int(0.1 * len(dataset))
    # Adjust train_size to make sure train_size + val_size + test_size == len(dataset)
    train_size += len(dataset) - (train_size + val_size + test_size)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class HeadlineDataLoader:
    """
    Class to handle loading headlines, tickers, dates, and info in batches.

    :param str data_folder: The folder with ticker pickle files. Files are in the format {ticker}.pkl
    :param str headlines_file: The file of the csv file with [i, ticker, headline, date, %change] rows
    :return: the DataLoader object
    """
    def __init__(self, data_folder: str, headlines_file: str):
        self.data_folder = data_folder
        self.headlines_file = headlines_file
        self.headlines_df = pd.read_csv(headlines_file)
    
    def load_ticker(self, ticker: str, batch_size: int, trading_days_before: int = 0, trading_days_after: int = 1):
        """
        Load the headlines and associated data for a given ticker with batch size. If there is no data for the given headline 
        date, skips it
        """
        ticker_df_path = os.path.join(self.data_folder, ticker + ".pkl")
        data_df = pd.read_pickle(ticker_df_path)

        ticker_headlines = self.headlines_df.loc[self.headlines_df['stock'] == ticker]
        # if len(ticker_headlines) < batch_size:
        #     print("Cannot make a batch of size " + str(batch_size) + " out of " + str(len(ticker_headlines)) + " headlines for ticker " + ticker)
        #     return (None, None, None)
        
        ticker_headlines = ticker_headlines.sample(n=batch_size)
        # Problem: Ticker headline dates don't necessarily line up with trading dates
        dates = data_df.index.to_numpy()
        dates_unix = dates.astype('datetime64[s]').astype('int')
        headline_dates_unix = ticker_headlines.loc[:, "date"].to_numpy().astype('datetime64[s]').astype('int')
        
        # The indices of the nearest previous trading day in data_df of the headlines
        indices = np.searchsorted(dates_unix, headline_dates_unix, side='right')
        slices = list(map(lambda index: slice(index-trading_days_after, index+trading_days_after), indices))
        headlines = ticker_headlines["title"].tolist()
        dates = ticker_headlines["date"].tolist()
        data = list(zip(headlines, dates, slices))
        return (data, ticker, data_df)

        # Returns ([headline, date, slice], ticker, data)

import time

def clean_up_csv():
    data_folder = "data/NumericalData/"
    headlines_file = "data/analyst_ratings_5col_fixed_headlines.csv"
    files = os.listdir(data_folder)
    tickers = []
    for file in files:
        tickers.append(file.split(".")[0])
    
    df = pd.read_csv(headlines_file)
    clean_df = df[df['stock'].isin(tickers)]
    pd.to_pickle(clean_df, "data/cleaned_headlines.pkl")

def main():
    data_folder = "data/NumericalData/"
    headlines_file = "data/cleaned_headlines.pkl"

    train_loader, val_loader, test_loader = create_data_loaders(headlines_file, data_folder)
    t = iter(train_loader)
    
    start = time.time()
    iters = 20
    for i in range(iters):
        x = next(t)
        if len(x) > 7000:
            print("woah")
    end = time.time()
    print(end - start)
    print(str((end-start)/iters) + "ms per batch")
    # print(count)

if __name__ == "__main__":
    main()
    # clean_up_csv()        