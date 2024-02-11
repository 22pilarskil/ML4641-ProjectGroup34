import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
from pandas_market_calendars import get_calendar
import os
import random
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time

options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})

# Get all tickers in S&P500
tickers_and_names = []

for row in table.findAll('tr')[1:]:
    columns = row.find_all('td')
    
    ticker = columns[0].text.strip()
    name = columns[1].text.strip()

    # Check if ticker and name are the same, if true, extract name from the third column
    # This is a weird bug with the table structure
    if ticker == name:
        name = columns[2].text.strip()

    tickers_and_names.append({ticker, name})

for ticker, name in tickers_and_names:
    query_name = name.replace(' ', '+')
    
    url = 'https://www.reuters.com/site-search/?query=' + query_name + '&offset=0'

    driver.get(url)

    delay = 5

    try:
        results_container = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'search-results__sectionContainer__34n_c')))
    except TimeoutException:
        print("Loading took too much time!")

    time.sleep(2)

    
