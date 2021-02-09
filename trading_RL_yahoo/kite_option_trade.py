import os
import sys
import numpy as np
import logging
import warnings
import pandas as pd 
from time import sleep
from datetime import date, datetime, timedelta
import pytz
from kite_historical_order import kiteparse
import talib
from talib import MA_Type

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def logic (ohlc, exchange="NSE"):
    inputs = {"open":np.array(ohlc.open.values),
          "high": np.array(ohlc.high.values),
          "low": np.array(ohlc.low.values),
          "close": np.array(ohlc.close.values),
          "volume": np.array(ohlc.volume.values).astype(float)}

    try:
        sma8 = talib.SMA(inputs["close"], 8)
        sma21 = talib.SMA(inputs["close"], 21)

        rsi = talib.RSI(inputs["close"], timeperiod=15)
        sma_rsi = talib.SMA(rsi,8)

        #natr = talib.NATR(inputs["high"],inputs["low"],inputs["close"], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACDEXT(inputs["close"], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        ultosc = talib.ULTOSC(inputs["high"],inputs["low"],inputs["close"],timeperiod1=7, timeperiod2=14, timeperiod3=28)
        slowk, slowd = talib.STOCH(inputs["high"],inputs["low"],inputs["close"], 5, 3, 0, 3, 0)
        #obv = talib.OBV(inputs["close"], inputs["volume"])
        bop = talib.SMA(talib.BOP(inputs["open"], inputs["high"],inputs["low"],inputs["close"]),5)

        data = pd.DataFrame({ "sma_rsi":sma_rsi,"slowk": slowk, "slowd": slowd,"rsi":rsi, "ultosc": ultosc,
                              "macd": macd, "macdsignal": macdsignal,"macdhist":macdhist,# "obv":obv, "natr": natr,
                              "bop":bop, "sma8":sma8, "sma21":sma21,
                              })
    except Exception as e:
		    print(item + str(e) + " can't process")
    return data
	
def pattern(ohlc):
    inputs = {"open":np.array(ohlc.open.values),
          "high": np.array(ohlc.high.values),
          "low": np.array(ohlc.low.values),
          "close": np.array(ohlc.close.values),
          "volume": np.array(ohlc.volume.values).astype(float)}
    dict1 = talib.get_function_groups()
    pattern = dict1['Pattern Recognition']
    dict2 = {}

    dict1 ={}
    for item in pattern:
        value = eval("talib.{}(inputs['open'],inputs['high'], inputs['low'], inputs['close'])".format(item))
        dict1["{}".format(item)] = value
    return dict1
	
def main():

    kite = kiteparse()
	
    data112=["NIFTY2121114900PE","NIFTY2121115100PE", "NIFTY2121115100CE","NIFTY2121115200CE","NIFTY2121115000PE", "BANKNIFTY2121135500PE","BANKNIFTY2121136200CE" ]
    # data112 = list(pd.read_csv("ind_nifty50list.csv")["Symbol"])
                #[nifty, nifty bank,]
    #data_index = ["256265","260105"]
    tzinfo = pytz.timezone('Asia/Kolkata')
    list1 = []
    list2 = []
    dict1 = {}
    df = pd.DataFrame(columns = ["time","symbol","call_price","call","patterns"])

    #while datetime.now(tz= tzinfo).time() > time(9, 15, 0) and datetime.now(tz= tzinfo).time() < time(15, 30, 0):
    for it in range(0,375):        
        exchange = "NFO"
        for item in data112:
            try:
                ohlc, now = kite.read_data_backtest(item, '5minute',exchange = exchange,symbol=True, minute= it)
                #ohlc, now = kite.read_data_backtest(item, '10minute',exchange = exchange,symbol=True)
                ta1 = logic(ohlc, exchange)

                ta1 = pd.concat([ohlc, ta1], axis=1)
                quantity = 75

                if item not in list1 and ta1["volume"][-2:].mean() > 0.6 * ta1["volume"][-50:-2].max()  :
                    if  ta1["rsi"][-1:].values[0] >  ta1["sma_rsi"][-2:-1].values[0] and  ta1["close"][-1:].values[0] > ta1["open"][-1:].values[0] and ta1["sma_rsi"][-2:].mean() > 0.8* ta1["sma_rsi"][-4:-1].mean() and  ta1["close"][-1:].values[0] > 0.9 * ta1["high"][-1:].values[0] :

                        # kite.placeorder(item, exchange, "BUY", quantity)

                        df.loc[len(df.index)] = [now, item,ohlc["close"][-1:].values[0], "Rising", ta1["slowk"][-1:].values[0]]
                        print([now, item,ohlc["close"][-1:].values[0], "Rising",ta1["slowk"][-1:].values[0]])
                        list1.append(item)
                        try:
                            list2.remove(item)
                        except: 
                            continue
                        continue

                if  item in list1 :
                    if  ta1["rsi"][-1:].values[0] < ta1["sma_rsi"][-1:].values[0] and ta1["sma_rsi"][-2:].mean() < ta1["sma_rsi"][-3:-1].mean():

                        # kite.placeorder(item, exchange, "SELL", quantity)

                        df.loc[len(df.index)] = [now, item,ohlc["close"][-1:].values[0], "Falling below sma8",ta1["slowk"][-1:].values[0]]
                        print([now, item,ohlc["close"][-1:].values[0], "Falling below sma8",ta1["slowk"][-1:].values[0]])
                        try:
                            list1.remove(item)
                        except:
                            continue
                        list2.append(item)
                        continue

                    if   ta1["close"][-1:].values[0] < 0.93 * ta1["high"][-2:].max() and ta1["rsi"][-1:].values[0] < ta1["sma_rsi"][-1:].mean() :
          
                        # kite.placeorder(item, exchange, "SELL", quantity)

                        df.loc[len(df.index)] = [now, item,ohlc["close"][-1:].values[0], "Falling below 5%",ta1["slowk"][-1:].values[0]]
                        print([now, item,ohlc["close"][-1:].values[0], "Falling below 5%",ta1["slowk"][-1:].values[0]])
                        try:
                            list1.remove(item)
                        except:
                            continue
                        list2.append(item)
                        continue

            except Exception as e:
                print(item + e)
                continue
    df.to_csv("today_option.csv")                

if __name__ == '__main__':
	main()