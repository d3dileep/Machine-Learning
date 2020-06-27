from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from util import csv_to_dataset, history_points
from tech_ind_model import data_split
from trading_algo import trade
import os,sys,json
from datetime import date, timedelta

def save_dataset(symbol):
    api_key = 'Q6Z2WPSDXG6T0O3P'
    print(symbol)
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data = data.head(800)
    data.to_csv(f'./{symbol}.csv')

if __name__ == "__main__":
    symbol = sys.argv[1]
    data_list = [x for x in os.listdir('.') if x.endswith('.csv')]
    current_date = date.today().isoformat()   
    if sys.argv[1]+'.csv' not in data_list:
        save_dataset(sys.argv[1])
        days = 200
        data_split(symbol,days)
        trade(symbol+'.csv',days)
    else:
        days = 200
        data_split(symbol,days)
        trade(symbol+'.csv',days)