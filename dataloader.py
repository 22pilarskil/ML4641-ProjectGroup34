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
    def __init__(self, headlines_file, numerical_folder):
        self.numerical_folder = numerical_folder
        self.headlines_file = headlines_file
        self.data = pd.read_csv(headlines_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 0]
        
        # Encode label as one-hot vector
        label_dict = {"positive": 0, "negative": 1, "neutral": 2}
        label_encoded = label_dict[label]
        label_one_hot = np.eye(len(label_dict))[label_encoded]

        # Tokenize text
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label_one_hot, dtype=torch.float)
        }


def create_data_loaders(filename, tokenizer, batch_size=32, encoding=None, max_len=256):
    # Initialize dataset
    dataset = SentimentAnalysisDataset(filename=filename, tokenizer=tokenizer, encoding=encoding, max_len=max_len)
    
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

def main():
    data_folder = "data/NumericalData/"
    headlines_file = "data/analyst_ratings_5col_fixed_headlines.csv"
    x = HeadlineDataLoader(data_folder, headlines_file)
    start = time.time()
    files = os.listdir(data_folder)
    count = 0
    for file in files:
        ticker = file.split(".")[0]
        data = x.load_ticker(ticker, 32)
        if file.startswith("B"):
            break
        count += len(data) / len(data)
    end = time.time()
    print(end - start)
    print(count)

if __name__ == "__main__":
    main()
        