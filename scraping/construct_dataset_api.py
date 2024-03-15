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

api_key = 'VYBKNAJE7RMO66IU'

##api_key = os.environ.get('ALPHA_API_KEY')
##
##if api_key is None:
##    print('Please set ALPHA_API_KEY environment variable')
##    exit()

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

years = ['2023', '2022', '2021', '2020', '2019']

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
debug_output_every_n = 50

with open('output_dataset.csv', 'a', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    if output_file.tell() == 0:
        csv_writer.writerow(['ticker', 'headline', 'date'])  # Write the header row

    total_relevance_score = 0
    total_valid_articles = 0
    total_relevance_score_filtered = 0
    total_valid_articles_filtered = 0
                        
    for count, ticker in enumerate(tickers):
        for month in months_start_and_end:
            for i, year in enumerate(years):
                start_timestamp = year + month[0] + 'T0900'
                end_timestamp = year + month[1] + 'T1600'

                try:
                    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&limit=1000&tickers=' \
                                + ticker + '&time_from=' + start_timestamp + '&time_to=' + end_timestamp \
                                + '&apikey=' + api_key
                    r = requests.get(url)
                    data = r.json()

                    ticker_data = yf.Ticker(ticker)

                    if 'longName' in ticker_data.info:
                        name = ticker_data.info['longName'].split()[0]

                        # Check the length of the ticker and add exchange name if less than 3 characters
                        if len(ticker) < 3:
                            ticker_with_exchange = ticker_data.info['exchange'] + ':' + ticker
                        else:
                            ticker_with_exchange = ticker

                        company_mentions = [ticker_with_exchange, name]
                    else:
                        if len(ticker) < 3:
                            ticker_with_exchange = ticker_data.info['exchange'] + ':' + ticker
                        else:
                            ticker_with_exchange = ticker

                    if 'feed' in data:
                        filtered_headlines = []

                        for article in data['feed']:
                            relevance_score = -1
                            
                            if 'ticker_sentiment' in article:
                                for t in article['ticker_sentiment']:
                                    if t['ticker'].lower() == ticker.lower():
                                        relevance_score = float(t['relevance_score'])

                            if relevance_score >= 0:
                                total_relevance_score += relevance_score
                                total_valid_articles += 1
            
                            for keyword in company_mentions:
                                if 'title' in article:
                                    if keyword.lower() in article['title'].lower():
                                        if relevance_score >= 0:
                                            total_relevance_score_filtered += relevance_score
                                            total_valid_articles_filtered += 1
                                            
                                        headline_date = datetime.datetime.strptime(article['time_published'], "%Y%m%dT%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")

                                        # Write the row to CSV
                                        csv_writer.writerow([ticker, article['title'], headline_date])
                                        
                                        break                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

                        break
                    else:
                        if i == len(years) - 1:    
                            print('Failed to retrieve data')
                            print(url)
                            print(data)

                    time.sleep(delay)
                            
                except Exception as e:
                    print(f"An exception occurred: {str(e)}")

        if count % debug_output_every_n == 0:
            if total_valid_articles > 0:
                average_relevance_score = total_relevance_score / total_valid_articles
            else:
                average_relevance_score = 0

            # Calculate average relevance score for filtered articles
            if total_valid_articles_filtered > 0:
                average_relevance_score_filtered = total_relevance_score_filtered / total_valid_articles_filtered
            else:
                average_relevance_score_filtered = 0

            print("Average relevance score for all articles:", average_relevance_score)
            print("Average relevance score for filtered articles added to CSV:", average_relevance_score_filtered)

