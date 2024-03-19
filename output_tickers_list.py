import os
import pickle

numerical_data_dir = 'data/NumericalData'

filenames = os.listdir(numerical_data_dir)

tickers = []

for filename in filenames:
    if os.path.isfile(os.path.join(numerical_data_dir, filename)):
        ticker = filename[:-4]
        tickers.append(ticker)

with open('data/tickers_list.pkl', 'wb') as f:
    pickle.dump(tickers, f)
