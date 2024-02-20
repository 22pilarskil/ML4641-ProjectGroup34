import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

class DataLoader:
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
    
    def load_ticker(self, ticker: str, batch_size: int):
        """
        Load the headlines and associated data for a given ticker with batch size. If there is no data for the given headline 
        date, skips it
        """
        ticker_df_path = os.path.join(self.data_folder, ticker + ".pkl")
        data_df = pd.read_pickle(ticker_df_path)

        ticker_headlines = self.headlines_df.loc[self.headlines_df['stock'] == ticker]
        if len(ticker_headlines) < batch_size:
            raise "Cannot make a batch of size " + str(batch_size) + " out of " + str(len(ticker_headlines)) + " headlines for ticker " + ticker
        
        ticker_headlines = ticker_headlines.sample(n=batch_size)
        min_date = ticker_headlines.min().loc["date"]
        utc = datetime.fromisoformat(min_date).timestamp()
        print(utc)
        return

        # Returns [ticker, headline, date, datapoints]



def main():
    data_folder = "pickled/"
    headlines_file = "data/analyst_ratings_5col_fixed_headlines.csv"
    x = DataLoader(data_folder, headlines_file)
    x.load_ticker("A", 1829)

if __name__ == "__main__":
    main()
        