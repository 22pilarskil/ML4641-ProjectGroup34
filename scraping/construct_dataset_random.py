import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
from pandas_market_calendars import get_calendar
import os
import random
import pickle
import csv
import time

api_key = os.environ.get('ALPHA_API_KEY')

if api_key is None:
    print('Please set ALPHA_API_KEY environment variable')
    exit()

##resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
##soup = bs.BeautifulSoup(resp.text, 'lxml')
##table = soup.find('table', {'class': 'wikitable sortable'})

# Get all tickers in S&P500
##tickers = []
##
##for row in table.findAll('tr')[1:]:
##    ticker = row.findAll('td')[0].text
##    tickers.append(ticker.strip())

tickers = []

with open('../data/tickers_list.pkl', 'rb') as f:
    tickers = pickle.load(f)

nyse = get_calendar('XNYS')

year = '2022'

months_start_and_end = [
    ('0101', '0131'),
    ('0201', '0228'),
    ('0301', '0331'),
    ('0401', '0430'),
    ('0501', '0531'),
    ('0601', '0630'),
    ('0701', '0731'),
    ('0801', '0831'),
    ('0901', '0930'),
    ('1001', '1031'),
    ('1101', '1130'),
    ('1201', '1231')
]

delay = 2

with open('output_dataset.csv', 'a', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    if output_file.tell() == 0:
        csv_writer.writerow(['ticker', 'headline', 'date'])  # Write the header row

    for ticker in tickers:
        for month in months_start_and_end:
            start_timestamp = year + month[0] + 'T0900'
            end_timestamp = year + month[1] + 'T1600'

            try:
                url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=' \
                            + ticker + '&time_from=' + start_timestamp + '&time_to=' + end_timestamp \
                            + '&apikey=' + api_key
                r = requests.get(url)
                data = r.json()

                ticker_data = yf.Ticker(ticker)

                if 'longName' in ticker_data.info:
                    name = ticker_data.info['longName'].split()[0]

                    company_mentions = [ticker, name]
                else:
                    company_mentions = [ticker]

                if 'feed' in data:
                    filtered_headlines = []

                    for article in data['feed']:
                        for keyword in company_mentions:
                            if keyword.lower() in article['title'].lower():
                                csv_writer.writerow([ticker, article['title'], article['time_published']])
                                break
                else:
                    print('Failed to retrieve data')
                    print(url)
                    print(data)

                time.sleep(delay)
                            
            except Exception as e:
                print(f"An exception occurred: {str(e)}")

##year = 2022
##
##trading_days = nyse.valid_days(start_date=f'2024-02-19', end_date=f'2024-02-23')
##
##for day in trading_days:
##    day_str = str(day)
##
##    year_month_day = day_str[:4] + day_str[5:7] + day_str[8:10]
##    
##    open_timestamp = year_month_day + 'T0930'
##    close_timestamp = year_month_day + 'T1600'
##
##    for ticker in tickers:
##        url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=' \
##                + ticker + '&time_from=' + open_timestamp + '&time_to=' + close_timestamp \
##                + '&apikey=' + api_key
##        r = requests.get(url)
##        data = r.json()
##
##        print(data)
##
##        ticker_data = yf.Ticker(ticker)
##
##        if 'longName' in ticker_data.info:
##            name = ticker_data.info['longName'].split()[0]
##
##            company_mentions = [ticker, name]
##        else:
##            company_mentions = [ticker]
##
##        headline_data = []
##
##        if 'feed' in data:
##            possible_headlines = []
##
##            for article in data['feed']:
##                for keyword in company_mentions:
##                    if keyword.lower() in article['title'].lower():
##                        possible_headlines.append({'ticker': ticker, 'headline': article['title'],
##                                                           'date': article['time_published']})
##                        break
##
##            selected_headline_data = random.choice(possible_headlines)
##
##            article_date = pd.to_datetime(article['time_published'], format='%y-%m-%dT%H%M')
##
##            # Get the closing price of the trading day before the article date
##            previous_trading_day = nyse.valid_days(start_date=article_date - pd.DateOffset(days=1), end_date=article_date)
##
##            print(previous_trading_day)
##            
##            if previous_trading_day:
##                previous_trading_day = previous_trading_day[-1]  # Get the last trading day
##                close_price_before = ticker_data.history(start=previous_trading_day, end=previous_trading_day + pd.Timedelta(days=1))['Close'].iloc[0]
##            else:
##                # Handle case when there is no previous trading day (e.g., Monday)
##                close_price_before = None
##
##            # Get the closing price of the trading day after the article date
##            next_trading_day = nyse.valid_days(start_date=article_date, end_date=article_date + pd.DateOffset(days=1))
##            if next_trading_day:
##                next_trading_day = next_trading_day[0]  # Get the first trading day
##                close_price_after = ticker_data.history(start=next_trading_day, end=next_trading_day + pd.Timedelta(days=1))['Close'].iloc[0]
##            else:
##                # Handle case when there is no next trading day (e.g., Friday)
##                close_price_after = None

            
