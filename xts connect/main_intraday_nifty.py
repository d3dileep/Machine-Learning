import pymongo
import json
import pytz
import requests
import pandas as pd
import numpy as np
import io
import math
from time import sleep
from datetime import date, datetime, timedelta, time
from xts_class2 import XTS_parse, XTS_order
from strategy2 import nif_buy_ce, nif_buy_pe
import configparser

def main(item):
    cfg = configparser.ConfigParser()
    cfg.read('token_nif.ini')
    xts= XTS_parse(token=cfg.get('datatoken', 'token'), userID=cfg.get('datauser', 'user'), isInvestorClient=True)
    quantity, dt = xts.get_id(item)
    xts_o = XTS_order(token=cfg.get('ordertoken', 'token'), userID=cfg.get('orderuser', 'user'), isInvestorClient=True)

    exchange = "NSEFO"
    tzinfo = pytz.timezone('Asia/Kolkata')
    underlying_mapping = {'NIFTY 50': 'NIFTY', 'NIFTY BANK': 'BANKNIFTY'}

    tradable_buy = []
    traded = []
    strikes = []
    dict1 = {}
    lot=1
    fo_instr_list = xts.get_instr_list()
    all_expiries = sorted(list(set([datetime.strptime(str(x)[4:10], '%y%m%d') for x in fo_instr_list[0].tolist()])))
    ## fetch nearest expiry if its friday, monday, or tuesday else next expiry
    if datetime.now(tz= tzinfo).weekday() in [0,1,4,5,6]:
        nearest_expiry = all_expiries[0]
    elif datetime.now(tz= tzinfo).weekday() in [2,3]:
        del all_expiries[0]
        nearest_expiry = all_expiries[0]

    #nearest_expiry = all_expiries[0]
    monthend_expiry = 'YES' if len([x for x in all_expiries if x.year == nearest_expiry.year and x.month == nearest_expiry.month]) == 1 else 'NO'
    print('starting calculation')

    print('get holding response',xts_o.get_holding())
    update_time = datetime.now(tz= tzinfo)
    df, _ = xts.read_data(item, 60,"NSECM", prime=True)
    day_delta = df['date'][-48:-47].dt.date.values[0]
    while datetime.now(tz= tzinfo).time() > time(9, 15, 0) and datetime.now(tz= tzinfo).time() < time(15, 30, 0):
        try:
            now = datetime.now(tz= tzinfo)

            if  now >= update_time:
                tradable_buy = []
                df, _ = xts.read_data(item, 60,"NSECM", prime=True)
                day_delta = df['date'][-48:-47].dt.date.values[0]
                if abs(df["open"][-1:].values[0] - df["close"][-2:-1].values[0]) >0.0039*df["close"][-2:-1].values[0] and datetime.now(tz= tzinfo).time() < time(9, 30, 0):
                    print("Gap Opening, sleeping till 09:30")
                    sleep((datetime(datetime.now(tz= tzinfo).year,datetime.now(tz= tzinfo).month,datetime.now(tz= tzinfo).day,9, 30) - datetime.now(tz= tzinfo).replace(tzinfo=None)).total_seconds())
                    if "buy CE" not in strikes:
                      tradable_buy.extend( nif_buy_ce(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                    if "buy PE" not in strikes:
                      tradable_buy.extend( nif_buy_pe(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                    update_time = now + timedelta(minutes=75)
                    print(now)
                else:
                    if "buy CE" not in strikes:
                      tradable_buy.extend( nif_buy_ce(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                    if "buy PE" not in strikes:
                      tradable_buy.extend( nif_buy_pe(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                    update_time = now + timedelta(minutes=75)
                    print(now)
                print("BUY",tradable_buy)

            if tradable_buy and datetime.now(tz= tzinfo).time() >= time(9, 25, 0) and datetime.now(tz= tzinfo).time() <= time(15, 10, 0):
                  for spot in tradable_buy :
                        sleep(1)
                        close_price = xts.get_latest_price(spot["symbol"])
                        if datetime.now(tz= tzinfo).second == 0:
                            print(datetime.now(tz= tzinfo).time(), spot["symbol"], close_price)
                        if close_price >= spot["entry_p"] :
                            print({"symbol":spot["symbol"], "entry_p":close_price, "action":spot['action'],"target":spot["target"],"stoploss":spot["stoploss"],"lot":lot})
                            xts_o.place_order(data,quantity,spot["symbol"],'BUY',lot)
                            traded.append({"time": now,"symbol":spot["symbol"], "entry_p":close_price, "action": spot['action'],"target":spot["target"],"stoploss":spot["stoploss"],"lot":lot})
                            strikes.append(spot['action'])
                            tradable_buy.remove(spot)
                            dict1[spot["symbol"]] = close_price
                # exit the entered positions
            if traded:
                    for position in traded:
                        sleep(1)
                        close_price = xts.get_latest_price(position["symbol"])
                        if datetime.now(tz= tzinfo).second in range(2):
                            print(datetime.now(tz= tzinfo).time(), position["symbol"], close_price)
                        if close_price > 1.1* dict1[position["symbol"]]:
                            position["stoploss"] = 1.05 * position["stoploss"]
                        if  position["target"] <= close_price:
                            print(position["symbol"], close_price, "exit bought")
                            xts_o.exit_order(data,quantity,position["symbol"],1)
                            position["lot"] -=1
                            position["target"] = xts.roundoff(1.2 *position["target"])
                            position["stoploss"] = xts.roundoff(0.8 *position["target"])
                            if position["lot"]==0:
                                traded.remove(position)
                                if "buy CE" in strikes:
                                  tradable_buy.extend( nif_buy_ce(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                                if "buy PE" in strikes:
                                  tradable_buy.extend( nif_buy_pe(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                                print(now,"\n","BUY",tradable_buy)
                                strikes.remove(position["action"])

                        if  datetime.now(tz= tzinfo).time() >= time(15, 15, 0):
                            print(position["symbol"], close_price, "stoploss bought")
                            xts_o.exit_order(data,quantity,position["symbol"],position["lot"])
                            position["lot"] = 0
                            if position["lot"]==0:
                                traded.remove(position)
                                strikes.remove(position["action"])

                        if position["stoploss"] >= close_price:
                            print(position["symbol"], close_price, "stoploss bought")
                            xts_o.exit_order(data,quantity,position["symbol"],position["lot"])
                            position["lot"] = 0
                            if position["lot"]==0:
                                traded.remove(position)
                                if "buy CE" in strikes:
                                  tradable_buy.extend( nif_buy_ce(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))
                                if "buy PE" in strikes:
                                  tradable_buy.extend( nif_buy_pe(item,nearest_expiry,monthend_expiry,exchange, xts, day_delta))                                
                                print(now,"\n","BUY",tradable_buy)
                                strikes.remove(position["action"])

        except Exception as e:
          print(e)

if __name__ == "__main__":
    main('NIFTY')
