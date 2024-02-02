import json
import requests
from datetime import datetime, timedelta
import csv
import more_itertools
import copy
import time
from decimal import *
import yfinance as yf

def get_bars(ticker, verbose=False):
    try: 
        bars = yf.Ticker(ticker).history('max')
        return bars
    except KeyError:
        if verbose:
            print('Not enough data for ticker ' + ticker)
        return None
    except AssertionError:
        if verbose:
            print('Stock may be delisted, no data found for ticker ' + ticker)
        return None

def extractBatch(csv_gen, verbose=False):
    line = next(csv_gen, None)
    ## If this line has already added the % change, ignore the batch and skip to next
    while line is not None and len(line) > 4:
        line = next(csv_gen, None)
    
    # End of csv_gen
    if line is None:
        return None
    
    batch = [line]
    [ind, _, _, ticker] = line
    if verbose:
        print("starting batch on line " + str(ind) + " for ticker: " + ticker)

    # Extract all lines of the csv file that (are consecutive and) have the same ticker
    while True:
        line = csv_gen.peek(None)
        if line is None: break
        elif len(line) < 4:
            raise IndexError("Line doesn't have at least 4 elements: " + str(line))
        
        # Don't want to dip into next batch of tickers
        if ticker != line[3]:
            break

        line = next(csv_gen)
        batch.append(line)
    
    return batch

# csv_lines is a 2d array, rows are lines in the csv file and row, columns are each comma separated value
def writeToCSV(csv_lines, flag="a"):
    with open("updated.csv", flag, encoding='utf-8', newline='') as myfile:
        csvWriter = csv.writer(myfile, delimiter=',')
        csvWriter.writerows(csv_lines)

def find_indexes(bars, date, verbose=False):
    try:
        ind = bars.index.get_loc(date)
        if ind == 0 or ind == len(bars) - 1:
            if verbose: print('failed to find surrounding indexes because the bars started or ended on that date')
            return None, None
        return [ind-1, ind+1]
    except KeyError:
        # Headline occurred on a non-trading day (weekend, holiday, or given bars just don't have the info)
        date_obj = datetime.fromisoformat(date)

        # Check the previous 5 days and see if any are a trading day. If not then it's just not in the bars.
        for i in range(1, 6):
            try:
                prev = (date_obj - timedelta(days = i)).date().isoformat()
                ind = bars.index.get_loc(prev)
                return[ind, ind+1]
            except KeyError:
                continue
        
        if verbose: print('failed to find indexes because no nearby bar date')
        return None, None

# Original csv had weird newlines and spacings in it. This consolidates them.
def filter_csv(csv_gen):
    new_csv = []
    nextLine = next(csv_gen, None)
    while nextLine is not None:
        while len(nextLine) < 4:
            followingLine = next(csv_gen)
            if len(followingLine) == 0: continue
            nextLine[-1] = nextLine[-1].strip() + followingLine.pop(0).strip()
            nextLine.extend(followingLine)
        
        # A couple of them have this and it's weird
        if "➡️" in nextLine[1]:
            nextLine[1] = nextLine[1].split("➡️", 1)[0]
        
        nextLine[1] = nextLine[1].strip()
        new_csv.append(nextLine)
        nextLine = next(csv_gen, None)
    
    return more_itertools.peekable(new_csv)

def add_percent_change():
    starting_at_ticker = "A"
    getcontext().prec = 5 # Precision of decimal points
    with open("analyst_ratings_processed.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=",")
        gen = more_itertools.peekable(reader)
        gen = filter_csv(gen)
        first_line = next(gen)
        first_line.append('percent change')
        if starting_at_ticker == "A": writeToCSV([first_line])
        has_processed_starting = False

        # Work by each ticker
        while True:
            batch = extractBatch(gen, True)
            if batch is None: 
                break
            elif batch[0][3] != starting_at_ticker and not has_processed_starting:
                continue
            has_processed_starting = True

            # Retrieve bars from api
            [_, _, _, ticker] = batch[0]
            bars = get_bars(ticker)
            
            updated_batch = []
            for csv_line in batch:
                [ind, headline, date, ticker] = csv_line
                
                # Find the previous trading day and next trading day
                [prev_idx, next_idx] = find_indexes(bars, date) # Need this because headlines can appear on non-trading days
                if prev_idx is None:
                    continue
                
                day_before_close = bars.iloc[prev_idx].loc['Close']
                day_after_close = bars.iloc[next_idx].loc['Close']
                
                # Calculate percent change and add it to the batch
                percent_change = Decimal(100.0 * (day_after_close - day_before_close)) / Decimal(day_before_close)
                updated_batch.append([ind, headline, date, ticker, percent_change])
            
            writeToCSV(updated_batch)


if __name__ == "__main__":
    add_percent_change()