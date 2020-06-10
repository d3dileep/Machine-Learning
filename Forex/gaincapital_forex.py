import pandas as pd
import time
import pytz
import argparse
from datetime import datetime
import os
import sys

from gcapi import GCapiClient

api = GCapiClient(username='DC254061', password='trade123', appkey='S.Horsley')

buy = {}
loss = []
quantity = {}
tz = pytz.timezone('UTC')

def get_historic_klines(market_id, quantity, interval, diff):
    ''' Returns the open, close, high and high_past 
    values in the given interval.
    market_id: Id of the market to get the data.
    quantity: no of records to retrive in the given interval.
    interval: type of interval (HOUR, MINUTE, DAY).
    diff: actual time interval b/w records.
    '''

    temp_data = api.get_ohlc(market_id=market_id, num_ticks=quantity * diff, interval=interval,span=1)
    
    data=api.get_prices(market_id=market_id, num_ticks=quantity*diff, from_ts=None, to_ts=None, price_type=None)
    last_timestamp_data=data['PriceTicks'][-1]
    #print(last_timestamp_data)
    timestamp=last_timestamp_data['TickDate']
    time=datetime.fromtimestamp(int(timestamp[6:-2])/1000)
    price=last_timestamp_data['Price']
    #print(time,price,sep='\t')
    
    #print(temp_data.tail(1))
    

    return temp_data.iloc[-1,1], price, temp_data.iloc[-1,4]


def Main():
    marketss = ['USD/JPY', 'EUR/USD', 'USD/CHF', 'USD/CAD', 'XAU/USD']
    interval = 'MINUTE'  # quantity type
    Quantity = 100  # no fo records
    diff = 1  # diff in interval
    
    ids = []
    for market in marketss:
        market = market.upper()
        print(market)
        for i in api.get_market_info(market_name=market, get=None)['Markets']:
            ids.append(i['MarketId'])
    list_of_symbols = ids  # Symbols to be traded
    print(ids)
    quantity_1 = 5  # any value between 1-4 : 1 =100%, 2=50%, 3 = 33%, 4 = 25%, 5 = 20% and so on...
    max_amount = 10000  # Maximum authorized amount
    loss_limit = -50  # Maximum loss limit to terminate the trading in dollar
    buy_percent = 0.0001  # percent at which it should buy, currently 0.15% = 0.15/100 = 0.0015
    sell_percent = 0.0006  # percent at which it should sell, currently 0.1%
    loss_percent = -0.0003  # stop loss if price falls, currently -0.3%
    transaction = 150  # number of maximum transactions
    buy_range = 0.0005  # allowed buy upto, currently 1.5%
    sleep_time = 10  # according to candle interval 15 for 5 MINUTE, 30 for 30 MINUTE, 45 for 1 HOUR
    spent_amount = 0
    count = 0
    buy_open = []  # to avoid multiple buy at same candle
    high = {}

    while True:  # USDT or BTC

        try:
            for symbol in list_of_symbols:
                #print('market_id: ', symbol)
                open1, close, prev_close = get_historic_klines(symbol, Quantity, interval, diff)

                #print(close)
                symbol = str(symbol)
                if open1 not in buy_open:
                    if (close >= (1 + buy_percent) * prev_close) and (symbol not in buy.keys()) and prev_close > open1:
                        if spent_amount <= max_amount:
                            count += 1
                            quantity[symbol] = (max_amount / (quantity_1 * close))
                            quantity1 = quantity[symbol]
                            buy_open.append(open1)
                            # high[symbol] = high1

                            spent_amount += close * quantity1
                            buy[symbol] = close
                            print('Bought ' + symbol + ' at ' + str(close))

                            df1 = pd.DataFrame({'Datetime': [datetime.now(tz)], 'Symbol': [symbol], 'Buy/Sell': ['Buy'],
                                                'Quantity': [quantity1], 'Price': [close], 'Profit/loss': [0]})
                            df1['Datetime'] = df1['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                            if not os.path.isfile('Binance.csv'):
                                df1.to_csv('Binance.csv', index=False)
                            else:
                                df1.to_csv('Binance.csv', index=False, mode='a', header=False)

                if symbol in buy:
                    # print(close, high[symbol], high1)
                    if (close >= buy[symbol] * (1 + sell_percent)) or (close <= (1 + loss_percent) * buy[symbol]):

                        #if close <= high[symbol] * (1 + sell_percent):
                        #    print('high limit')
                        profit = close - buy[symbol]
                        max_amount += profit
                        quantity1 = quantity[symbol]
                        spent_amount -= quantity1 * buy[symbol]
                        total_profit = profit * quantity1
                        print("SELL " + symbol + " at " + str(close))
                        print("Profit made " + str(total_profit))

                        df2 = pd.DataFrame({'Datetime': [datetime.now(tz)], 'Symbol': [symbol], 'Buy/Sell': ['Sell'],
                                            'Quantity': [quantity1], 'Price': [close], 'Profit/loss': [total_profit]})
                        df2['Datetime'] = df2['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                        df2.to_csv('Binance.csv', index=False, mode='a', header=False)

                        loss.append(total_profit)
                        if count >= len(list_of_symbols):
                            loss.pop(0)
                        buy.pop(symbol)  # Removing the sold symbol

                if (loss_limit > sum(loss)) or (count > int(transaction)):
                    print("Quitting....")
                    raise SystemExit

            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Total Profit " + str(sum(loss)))
            break

        except SystemExit:
            print("Exit")
            print("Total Profit " + str(sum(loss)))
            break
        time.sleep(3)

if __name__ == '__main__':
    Main()
