import yfinance as yf
import numpy as np
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    # Fetch historical market data
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    # Calculate Market Capitalization
    df['Market Cap'] = df['Close'] * stock.info['sharesOutstanding']

    # Calculate daily Volatility as the range between High and Low prices
    df['Volatility'] = df['High'] - df['Low']

    # Calculate the Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Apply log transformation to Close, Volume, and Market Cap
    df['Log Close'] = np.log(df['Close'] + 1)  # Adding 1 to avoid log(0)
    df['Log Volume'] = np.log(df['Volume'] + 1)  # Adding 1 to avoid log(0)
    df['Log Market Cap'] = np.log(df['Market Cap'] + 1)  # Adding 1 to avoid log(0)

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


    return df[['Log Close', 'Log Volume', 'Log Market Cap', 'Volatility', 'RSI', 'Percent Change''SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line']]

# Example usage
ticker_data = get_stock_data('AAPL', '2020-01-01', '2021-01-01')
print(ticker_data[-10:])
ticker_data = get_stock_data('AMD', '2020-01-01', '2021-01-01')
print(ticker_data[-10:])
