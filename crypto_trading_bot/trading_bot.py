import pandas as pd
import time
import dateparser
import pytz
import argparse
from datetime import datetime
from binance.client import Client
import os
# import matplotlib.pyplot as plt
# import mpl_finance

api_key = ""
api_secret = ""

client = Client(api_key, api_secret)
tz = pytz.timezone('UTC')
buy = {}
loss = []
quantity = {}

def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    d = dateparser.parse(date_str)
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)
    return int((d - epoch).total_seconds() * 1000.0)

def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    For clarification see document or mail d3dileep@gmail.com
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms


def get_historical_klines(symbol, interval, start_str, end_str=None):
    """Get Historical Klines from Binance
    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC", "1 Dec, 2017"
    :param symbol: Name of symbol pair e.g BNBBTC
    :param interval: Biannce Kline interval
    :param start_str: Start date string in UTC format
    :param end_str: optional - end date string in UTC format
    :return: list of Open High Low Close Volume values
    """
    output_data = []
    limit = 50
    timeframe = interval_to_milliseconds(interval)
    start_ts = date_to_milliseconds(start_str)
    end_ts = None
    if end_str:
        end_ts = date_to_milliseconds(end_str)

    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        temp_data = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts
        )
        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_data):
            symbol_existed = True
        if symbol_existed:
            output_data += temp_data
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            start_ts += timeframe
        idx += 1
        if len(temp_data) < limit:
            break
        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)

    return output_data


def get_historic_klines(symbol, start, end, interval):
    klines = get_historical_klines(symbol, interval, start, end)
    ochl = []
    for kline in klines:
        time1 = int(kline[0])
        open1 = float(kline[1])
        low = float(kline[2])
        high = float(kline[3])
        close = float(kline[4])
        volume = float(kline[5])
        ochl.append([time1, open1, close, high, low, volume])
    '''
    fig, ax = plt.subplots()
    mpl_finance.candlestick_ochl(ax, ochl, width=1)
    ax.set(xlabel='Date', ylabel='Price', title='{} {}-{}'.format(symbol, start, end))
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    '''
    return ochl[-1][1], ochl[-1][2]


def Main():
    list_of_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Symbols to be traded
    quantity_1 = 1 # any value between 1-4 : 1 =100%, 2=50%, 3 = 33%, 4 = 25%, 5 = 20% and so on...
    max_amount = 10000  # Maximum authorized amount
    loss_limit = -3  # Maximum loss limit to terminate the trading
    buy_percent = 0.05  # percent at which it should buy
    sell_percent = 0.03 # percent at which it should sell
    loss_percent = -0.1 # stop loss if price falls 1%
    transaction = 150 # number of maximum transactions
    count = 0
    while True:

        try:
            client.get_deposit_address(asset='USDT')  # USDT or BTC

            for symbol in list_of_symbols:

                open1, close = get_historic_klines(symbol, "1 hours ago UTC", "now UTC", Client.KLINE_INTERVAL_1MINUTE)

                symbol = str(symbol)

                if (((open1 / close) - 1) * 100 >= buy_percent) and (symbol not in buy.keys()):

                    count += 1

                    if count > int(transaction):
                        print(
                            "Exceeded Number of transactions, Wont buy any stock. Waiting until bought stocks to be "
                            "sold")
                        continue
                    quantity[symbol] = int(max_amount / (quantity_1 * close))
                    '''
                    client.order_limit_buy(
                        symbol=symbol,
                        quantity=quantity[symbol],
                        price=close  # comment this line to buy at market price
                        )
                    '''
                    buy[symbol] = close
                    print('Bought ' + symbol + ' at ' + str(close))

                    df1 = pd.DataFrame({'Datetime': [datetime.now(tz)], 'Symbol': [symbol], 'Buy/Sell': ['Buy'],
                                        'Quantity': [quantity[symbol]], 'Price': [close], 'Profit/loss': [0]})
                    df1['Datetime'] = df1['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                    if not os.path.isfile('Binance.csv'):
                        df1.to_csv('Binance.csv', index=False)

                    else:
                        df1.to_csv('Binance.csv', index=False, mode='a', header=False)

                if symbol in buy:
                    if ((((close / buy[symbol]) - 1) * 100 >= sell_percent) or (((close / buy[symbol]) - 1) * 100 <= loss_percent)):
                        '''
                        client.order_limit_sell(
                          symbol=symbol,
                          quantity=quantity[symbol],
                          price=close  # comment this line to sell at market price
                          )
                        '''
                        profit = close - buy[symbol]

                        print("Sold " + symbol + " at " + str(close))
                        print("Profit made " + str(profit))

                        df2 = pd.DataFrame({'Datetime': [datetime.now(tz)], 'Symbol': [symbol], 'Buy/Sell': ['Sell'],
                                        'Quantity': [quantity[symbol]], 'Price': [close], 'Profit/loss': [profit]})
                        df2['Datetime'] = df2['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                        df2.to_csv('Binance1.csv', index=False, mode='a', header=False)

                        loss.append(profit)
                        buy.pop(symbol)  # Removing the sold symbol

                if (loss_limit > sum(loss)) or (count > int(transaction)):
                    print("Quitting....")
                    raise SystemExit

            time.sleep(15)

        except KeyboardInterrupt:
            print("Total Profit " + str(sum(loss)))
            break

        except SystemExit:
            print("Exit")
            print("Total Profit " + str(sum(loss)))
            break


if __name__ == "__main__":
    Main()
