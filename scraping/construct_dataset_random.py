import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
from pandas_market_calendars import get_calendar
import os
import random

api_key = os.environ.get('ALPHA_API_KEY')

if api_key is None:
    print('Please set ALPHA_API_KEY environment variable')
    exit()

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})

# Get all tickers in S&P500
tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker.strip())

nyse = get_calendar('XNYS')

year = 2022

trading_days = nyse.valid_days(start_date=f'2022-01-01', end_date=f'2022-12-31')

for day in trading_days:
    day_str = str(day)

    year_month_day = day_str[:4] + day_str[5:7] + day_str[8:10]
    
    open_timestamp = year_month_day + 'T0930'
    close_timestamp = year_month_day + 'T1600'

    for ticker in tickers:
        url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=' \
                + ticker + '&time_from=' + open_timestamp + '&time_to=' + close_timestamp \
                + '&apikey=' + api_key
        r = requests.get(url)
        data = r.json()

        ticker_data = yf.Ticker(ticker)

        if 'longName' in ticker_data.info:
            name = ticker_data.info['longName'].split()[0]

            company_mentions = [ticker, name]
        else:
            company_mentions = [ticker]

        headline_data = []

        if 'feed' in data:
            for article in data['feed']:
                possible_headlines = []
                
                for keyword in company_mentions:
                    if keyword.lower() in article['title'].lower():
                        possible_headlines.append({'headline': article['title'],
                                                           'date': article['time_published']})
                        
                        break

                headline_data.append(random.choice(possible_headlines))
