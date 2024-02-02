import os
import requests

api_key = os.environ.get('ALPHA_API_KEY')

if api_key is None:
    print('Please set ALPHA_API_KEY environment variable')
    exit()

time_from = '20240202T0930'
time_to = '20240202T1600'

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&time_from=' + time_from + '&time_to=' + time_to + '&apikey=' + api_key
r = requests.get(url)
data = r.json()

company_mentions = ['AAPL', 'Apple']

headline_data = []

for article in data['feed']:
    for keyword in company_mentions:
        if keyword.lower() in article['title'].lower():
            headline_data.append({'headline': article['title'],
                                  'date': article['time_published']})
            break

print(headline_data)
