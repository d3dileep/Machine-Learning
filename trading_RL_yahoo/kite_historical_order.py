import os
import sys
import numpy as np
import logging
import warnings
import pandas as pd 
from time import sleep
from datetime import date, datetime, timedelta
import pytz
from kiteconnect import KiteConnect
import talib
from talib import MA_Type

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class kiteparse:
    def __init__(self):
        self.kite = KiteConnect(api_key="2x2f2a8447agu452")
        print(self.kite.login_url())
        access_token = input("Enter token:") 
        data = self.kite.generate_session(access_token, api_secret="kfygfd0sgae19vnmsz49pnd1t44dacb1")
        self.kite.set_access_token(data["access_token"])
        data = self.kite.instruments()
        df = pd.DataFrame(data)[["instrument_token","exchange_token","tradingsymbol","exchange"]]
        df.to_csv("instruments_com.csv")
    def read_data_backtest(self, symbol_1, interval, exchange="NSE", symbol=True, minute=0):
        instrument = symbol_1
        if symbol:
            list_x = pd.read_csv("instruments_com.csv")
            dat = list_x.loc[list_x['tradingsymbol'] == symbol_1 , ['exchange','instrument_token']]
            instrument = dat.loc[dat['exchange'] == exchange, 'instrument_token'].iloc[0]
            
        from datetime import time
        tzinfo = pytz.timezone('Asia/Kolkata')

        now = datetime.now(tz= tzinfo)#tz= tzinf
        today = now.date()
        current_time = time(now.hour, now.minute, now.second)
      # print(current_time,today, today.weekday())


        if current_time < time(9,15,00) and today.weekday() == 0:
          current_time = time(9, 15, 00)
          to_d = datetime.combine(today-timedelta(days=3), current_time)+ timedelta(minutes = minute)

        elif current_time < time(9,15,00) and today.weekday() in range(5):
          current_time = time(9, 15, 00)
          to_d = datetime.combine(today-timedelta(days=1), current_time)+ timedelta(minutes = minute)

        elif current_time > time(15,31,00) and today.weekday() in range(5):
          current_time = time(9, 15, 00) 
          to_d = datetime.combine(today , current_time )+ timedelta(minutes = minute)

        elif today.weekday() == 5:
          current_time = time(9, 15, 00) 
          to_d = datetime.combine(today-timedelta(days=1), current_time) + timedelta(minutes = minute)

        elif today.weekday() == 6:
          current_time = time(9, 15, 00) 
          to_d = datetime.combine(today -timedelta(days=2), current_time )+ timedelta(minutes = minute)

        elif today.weekday() in range(5):
          to_d = datetime.combine(today, current_time)
        
        if interval == "2minute":
          period = timedelta(minutes=400*2 + 3550)
          from_d = to_d - period
        if interval == "5minute":
          period = timedelta(minutes=400*5 + 5550)
          from_d = to_d - period
        if interval == "10minute":
          period = timedelta(minutes=400*10 + 12550)
          from_d = to_d - period
        if interval == "15minute":
          period = timedelta(minutes=400*15 + 18550)
          from_d = to_d - period
        if interval == "30minute":
          period = timedelta(minutes=400*30 + 40550)
          from_d = to_d - period
        if interval == "60minute":
          period = timedelta(hours=2500)
          from_d = to_d - period
        if interval == "2hour":
          period = timedelta(hours=5000)
          from_d = to_d - period
        if interval == "3hour":
          period = timedelta(hours=8000)
          from_d = to_d - period
        if interval == "day":
          period = timedelta(days=399)
          from_d = to_d - period
        #from_d = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
        ohlcv = pd.DataFrame(self.kite.historical_data(instrument_token=instrument,from_date=from_d,to_date=to_d,interval=interval))
        return ohlcv, to_d
    def placeorder(self,item, exchange, action, quantity):
        try:    
            order_id = kite.place_order(tradingsymbol=item,
                                        exchange=exchange,
                                        transaction_type=action,
                                        quantity=quantity,
                                        variety="regular",
                                        order_type="MARKET",
                                        product="MIS")

            logging.info("Order placed. ID is: {}".format(order_id))
        except Exception as e:
            print(e)