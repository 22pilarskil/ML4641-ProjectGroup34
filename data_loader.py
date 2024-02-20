import os
import csv
import time

import yahoo_fin.stock_info as si
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

# Helper function to get stock data
def get_stock_data(ticker, start_date, end_date):
    # Fetch historical market data
    stock = yf.Ticker(ticker)
    earliest_start_date = '1900-01-01'  # Default early date
    try:
        # Fetch the stock's info which includes the IPO date
        ipo_date = stock.info.get('ipoStartDate', earliest_start_date)
        # Convert IPO date to datetime, or use the default early date
        if ipo_date == 'N/A' or ipo_date is None:
            start_date_dt = datetime.strptime(earliest_start_date, '%Y-%m-%d')
        else:
            start_date_dt = datetime.strptime(ipo_date, '%Y-%m-%d')
    except Exception as e:
        print(f"Error fetching IPO date for {ticker}: {e}")
        start_date_dt = datetime.strptime(earliest_start_date, '%Y-%m-%d')

    # Add 20 days to the determined start date
    start_date_dt += timedelta(days=28)

    # Ensure end_date is a datetime object
    if isinstance(end_date, str):
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    elif isinstance(end_date, datetime):
        end_date_dt = end_date
    else:
        raise ValueError("end_date must be a string or datetime object")

    # Ensure start_date_dt does not exceed end_date_dt
    if start_date_dt > end_date_dt:
        print(f"Adjusted start date {start_date_dt.strftime('%Y-%m-%d')} is after the end date {end_date_dt.strftime('%Y-%m-%d')}. Adjusting start date to match end date.")
        start_date_dt = end_date_dt

    start_date_dt -= timedelta(days=28)

    # Fetch historical market data using the adjusted start date
    df = stock.history(start=start_date_dt.strftime('%Y-%m-%d'), end=end_date_dt.strftime('%Y-%m-%d'))

    # try:
    #     # Fetch the stock's info which includes the IPO date
    #     ipo_date = stock.info.get('ipoStartDate', '1900-01-01')
    #     # In some cases, the IPO date might not be available or formatted differently,
    #     # so you set a default early date that's generally earlier than most IPOs.
    #     if ipo_date == 'N/A' or ipo_date is None:
    #         start_date = '1900-01-01'
    #     else:
    #         start_date = ipo_date
    # except Exception as e:
    #     print(f"Error fetching IPO date for {ticker}: {e}")
    #     start_date = '1900-01-01'  # Fallback to a very early start date

    # df = stock.history(start=start_date, end=end_date)
    #print(df)
    # try:
    #     if (stock.info['quoteType'] != "EQUITY" or not 'Close' in stock.info):
    #         return None
    # except:
    #     return None

    # Check if 'Close' data is available

    
    if 'Close' not in df.columns:
        print(f"No 'Close' data for {ticker}.")
        return None

    # Calculate Market Capitalization if possible
    try:
        shares_outstanding = stock.info.get('sharesOutstanding', np.nan)
        df['Market Cap'] = df['Close'] * shares_outstanding
    except Exception as e:
        df['Market Cap'] = np.nan
        print(f"Error calculating Market Cap for {ticker}: {e}")

    # Calculate daily Volatility as the range between High and Low prices
    df['Volatility'] = df['High'] - df['Low']

    # Ensure DataFrame is not empty
    if df.empty:
        print(f"No data available for {ticker}.")
        return None

    # Ensure there are enough data points for the RSI calculation
    rsi_period = 14
    if len(df) < rsi_period:
        print(f"Not enough data to calculate RSI for {ticker}. Need at least {rsi_period} days of data.")
        df['RSI'] = np.nan  # Set RSI to NaN for all rows if not enough data
        return df

    # Your RSI calculation logic...
    df['RSI'] = np.nan  # Initialize the RSI column
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the first RSI value based on the first 'rsi_period' days
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean().iloc[rsi_period-1]
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean().iloc[rsi_period-1]

    if avg_loss == 0:
        initial_rsi = 100
    else:
        rs = avg_gain / avg_loss
        initial_rsi = 100 - (100 / (1 + rs))

    # Set the initial RSI value
    df.at[df.index[rsi_period-1], 'RSI'] = initial_rsi

    # Calculate subsequent RSI values
    for i in range(rsi_period, len(df)):
        gain = delta.iloc[i] if delta.iloc[i] > 0 else 0
        loss = -delta.iloc[i] if delta.iloc[i] < 0 else 0
        avg_gain = (avg_gain * (rsi_period - 1) + gain) / rsi_period
        avg_loss = (avg_loss * (rsi_period - 1) + loss) / rsi_period

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        df.at[df.index[i], 'RSI'] = rsi

    # Apply log transformation to Close, Volume, and Market Cap
    # Adding 1 to avoid log(0) for each
    df['Log Close'] = np.log(df['Close'] + 1)
    df['Log Volume'] = np.log(df['Volume'] + 1)
    df['Log Market Cap'] = np.log(df['Market Cap'] + 1)

    if df.empty:
        print("DataFrame is empty.")
        return None  # or handle as appropriate
    
    # Ensure 'Close' column has valid data
    if not df['Close'].notna().any():
        print("No valid 'Close' data.")
        return None  # or handle as appropriate

    # Calculate daily percent change in Close price
    df['Percent Change'] = df['Close'].pct_change() * 100

    # SMA and EMA calculations
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD and Signal Line
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Ensure all missing values are explicitly set to NaN

    df.fillna(value=np.nan, inplace=True)
    

    return df[['Close', 'Volume', 'Market Cap', 'Log Close', 'Log Volume', 'Log Market Cap', 'Volatility', 'RSI', 'Percent Change', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line']]


# Function to save data to CSV
def save_to_pickle(data, ticker):
    filename = f"{ticker}.pkl"
    # Make sure the directory exists
    os.makedirs("Numerical_Data", exist_ok=True)
    data.to_pickle("Numerical_Data/" + filename)
    
    print(f"Data for {ticker} saved to {filename}")

# Assuming analyst_ratings_with_percent_change is a file containing the tickers
def save_to_csv(data, ticker):
    filename = f"{ticker}.csv"
    # Make sure the directory exists
    directory = "Numerical_Data_CSV"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    data.to_csv(file_path, index=False)
    print(f"Data for {ticker} saved to {filename}")

def load_tickers_from_file(file_path):

# The path to your CSV file
    csv_file_path = file_path

    # A set to store unique tickers
    tickers = set()

    # Open the CSV file with utf-8 encoding and read each row
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row) >= 4:  # Ensure there are at least 4 items in the row
                ticker = row[3]  # The ticker is the fourth item in the row
                tickers.add(ticker)  # Add the ticker to the set

# Print the list of unique stock tickers
    #print(list(tickers))            
    return (list(tickers))


# Main function to pull data for each ticker
def main():
    # try:
    #     with open('Numerical_Data/test_file.txt', 'w') as f:
    #         f.write('This is a test.')
    #     print("File written successfully.")
    # except PermissionError:
    #     print("Permission denied: Unable to write to the directory.")

    tickers = load_tickers_from_file('analyst_ratings_5col_fixed_headlines.csv')
    
    
    for ticker in tickers:
        #print(f"Fetching data for {ticker}")
        end_date = datetime.now()
        earliest_start_date = '1900-01-01'
        
        
        data = get_stock_data(ticker, earliest_start_date, end_date)
        if (not data is None):
            data.index = data.index.tz_localize(None)
            
            # Determine the actual start date for saving the data
            # This should consider the need for historical data for SMA calculations
            # Assuming get_stock_data already adjusts start_date based on IPO
            actual_start_date = max(data.index.min(), datetime.strptime('1900-01-01', '%Y-%m-%d')) + timedelta(days=28)

            # Now perform the comparison
            filtered_data = data[data.index >= actual_start_date]
            
            #save_to_pickle(filtered_data, ticker)
            save_to_csv(filtered_data, ticker)
        time.sleep(1)
    #print(data)

if __name__ == "__main__":
    main()

