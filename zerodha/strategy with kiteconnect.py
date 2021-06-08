import sys
import logging
import warnings
import pytz
import random as rnd
import numpy as np
import pandas as pd 
from time import sleep
from datetime import date, datetime, timedelta, time

from kiteconnect import KiteConnect
import pandas_ta as ta

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
        self.df = pd.DataFrame(data)[["instrument_token","exchange_token","tradingsymbol","exchange"]]


    def read_data_backtest(self, symbol_1, interval, exchange="NSE", symbol=True, days=0):
        instrument = symbol_1
        if symbol:
            dat = self.df.loc[self.df['tradingsymbol'] == symbol_1 , ['exchange','instrument_token']]
            instrument = dat.loc[dat['exchange'] == exchange, 'instrument_token'].iloc[0]
                
        tzinfo = pytz.timezone('Asia/Kolkata')

        now = datetime.now(tz= tzinfo)#tz= tzinf
        today = now.date()
        current_time = time(now.hour, now.minute, now.second)
      # print(current_time,today, today.weekday())

        to_d = datetime.combine(today, current_time))
        from_d = datetime.combine(today-timedelta(days=days), time(9, 15, 00)))
        #from_d = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
        ohlcv = pd.DataFrame(self.kite.historical_data(instrument_token=instrument,from_date=from_d,to_date=to_d,interval=interval))
        return ohlcv, to_d
    def placeorder(self,item, exchange, action, quantity):
        try:    
            order_id = self.kite.place_order(tradingsymbol=item,
                                        exchange=exchange,
                                        transaction_type=action,
                                        quantity=quantity,
                                        variety="regular",
                                        order_type="MARKET",
                                        product="MIS")

            logging.info("Order placed. ID is: {}".format(order_id))
        except Exception as e:
            print(e)
    def get_market_depth(self, item, exchange):
        """
        Fetch the market depth
        Args:
            instrument: Financial Instrument
        Returns:
            market depth value
        """
        dat = self.df.loc[self.df['tradingsymbol'] == item , ['exchange','instrument_token']]
        instrument = dat.loc[dat['exchange'] == exchange, 'instrument_token'].iloc[0]
        quote = self.kite.quote(instrument)
        return quote["{}".format(instrument)]
kite = kiteparse()
tzinfo = pytz.timezone('Asia/Kolkata')

list1 = []
list2 = []
dict1={}
data112 = ["SBICARD","TATASTEEL","RELIANCE","INFY","TATAMOTORS","SBIN","CIPLA","AXISBANK","WIPRO","MOTHERSUMI","ADANIPORTS"]
for item in data112:
    dict1["{}_buy".format(item)] =1

data = pd.DataFrame(columns = ["time","symbol","call_price","call","patterns"])
while datetime.now(tz= tzinfo).time() > time(9, 15, 0) and datetime.now(tz= tzinfo).time() < time(15, 30, 0):
#for it in range(0,22400, 55):        
    exchange = "NSE"
    for item in data112:
        #quote = kite.get_market_depth(item, exchange)
        #buy_sell= quote["buy_quantity"]/quote["sell_quantity"]

        df, now = kite.read_data_backtest(item, '15minute',exchange = exchange,symbol=True,days=10)
        df.ta.zscore(append=True)
        df.ta.sma(length=10, append=True)
        df.ta.dema(length=5,append=True)
        #df.ta.ha(append=True)
        df.ta.squeeze(lazybear=True, detailed=False, append=True)
        df.ta.stoch(append=True)
        #print(item)#,df.tail(),df.columns)
        #sleep(20)
        if item not in list1:
            if df["STOCHk_14_3_3"][-1:].values[0] > 1.05*  df["STOCHk_14_3_3"][-2:-1].values[0] and df["DEMA_5"][-1:].values[0] > 1.002*df["DEMA_5"][-2:-1].values[0] and df["DEMA_5"][-1:].values[0] > df["SMA_10"][-1:].values[0] and df["SQZ_20_2.0_20_1.5_LB"][-1:].values[0] > 1.001*  df["SQZ_20_2.0_20_1.5_LB"][-2:-1].values[0] and df["close"][-1:].values[0] > 0.994* df["high"][-1:].values[0]:
                dict1["{}_buy".format(item)] = df["close"][-1:].values[0]
                data.loc[len(df.index)] = [now, item,df["close"][-1:].values[0], "Rising",df["STOCHk_14_3_3"][-2:].values]
                print([str(now), item,df["close"][-1:].values[0], "Rising",df["STOCHk_14_3_3"][-2:].values])
                list1.append(item)
                #continue

        if item in list1:
            if df["STOCHk_14_3_3"][-1:].values[0] < 0.95*df["STOCHk_14_3_3"][-2:-1].values[0] or df["close"][-1:].values[0] < 0.995*dict1["{}_buy".format(item)] or df["close"][-1:].values[0] < 0.99*df["high"][-2:].max() :
                data.loc[len(df.index)] = [now, item,df["close"][-1:].values[0], "exit_bought", df["STOCHk_14_3_3"][-2:].values]
                print([str(now), item,df["close"][-1:].values[0], "exit_bought",df["STOCHk_14_3_3"][-2:].values])
                list1.remove(item)
                #continue
        if item not in list2:
            if df["STOCHk_14_3_3"][-1:].values[0] < 0.95*  df["STOCHk_14_3_3"][-2:-1].values[0] and df["DEMA_5"][-1:].values[0] < 0.998*df["DEMA_5"][-2:-1].values[0] and df["DEMA_5"][-1:].values[0] < df["SMA_10"][-1:].values[0]  and df["SQZ_20_2.0_20_1.5_LB"][-1:].values[0] < 0.999*  df["SQZ_20_2.0_20_1.5_LB"][-2:-1].values[0] and df["close"][-1:].values[0] < 1.003* df["low"][-1:].values[0]:
                dict1["{}_buy".format(item)] = df["close"][-1:].values[0]
                data.loc[len(df.index)] = [now, item,df["close"][-1:].values[0], "FALLING",df["STOCHk_14_3_3"][-2:].values]
                print([str(now), item,df["close"][-1:].values[0], "FALLING",df["STOCHk_14_3_3"][-2:].values])
                list1.append(item)
                #continue

        if item in list2:
            if df["STOCHk_14_3_3"][-1:].values[0] > 1.05*df["STOCHk_14_3_3"][-2:-1].values[0] or df["close"][-1:].values[0] > 1.005*dict1["{}_buy".format(item)] or df["close"][-1:].values[0] > 1.005*df["low"][-2:].min()  :
                data.loc[len(df.index)] = [now, item,df["close"][-1:].values[0], "exit_bought", df["STOCHk_14_3_3"][-2:].values]
                print([str(now), item,df["close"][-1:].values[0], "exit_bought",df["STOCHk_14_3_3"][-2:].values])
                list1.remove(item)
                #continue

        
data.to_csv("today_equity.csv")                 
