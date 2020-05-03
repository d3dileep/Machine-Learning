import time
import dateparser
import pytz
import matplotlib.pyplot as plt
import mpl_finance
import argparse
from datetime import datetime
from binance.client import Client

api_key = "iZHsmlsCReb9S6zVO05Vxy8ONQYK8J3CfshgNiRh3HlRShPULMj8EYBClftHBqi1"
api_secret = "4IHk54oeSmmoXGQqWNgi24SJ1uHaTSEBfN48nOhYex8ATFFOj2WoWZQfDFD0pzu1"

client = Client(api_key, api_secret)
address = client.get_deposit_address(asset='USDT')  # USDT or BTC


def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds

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

    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Biannce Kline interval
    :type interval: str
    :param start_str: Start date string in UTC format
    :type start_str: str
    :param end_str: optional - end date string in UTC format
    :type end_str: str

    :return: list of OHLCV values

    """

    # init our list
    output_data = []

    # setup the max limit
    limit = 500

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    # convert our date strings to milliseconds
    start_ts = date_to_milliseconds(start_str)

    # if an end time was passed convert it
    end_ts = None
    if end_str:
        end_ts = date_to_milliseconds(end_str)

    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
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
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)

    return output_data


def get_historic_klines(symbol, start, end, interval):
    klines = get_historical_klines(symbol, interval, start, end)
    ochl = []

    for kline in klines:
        # print(kline)
        time1 = int(kline[0])
        open1 = float(kline[1])
        Low = float(kline[2])
        High = float(kline[3])
        Close = float(kline[4])
        Volume = float(kline[5])

        ochl.append([time1, open1, Close, High, Low, Volume])

    fig, ax = plt.subplots()
    mpl_finance.candlestick_ochl(ax, ochl, width=1)
    ax.set(xlabel='Date', ylabel='Price', title='{} {}-{}'.format(symbol, start, end))
    plt.show()
    return ochl[-1][1], ochl[-1][2]


def buy_sell(args):
    list_of_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    while (True):

        for symbol in list_of_symbols:

            open1, close = get_historic_klines(symbol, "1 hours ago UTC", "now UTC", Client.KLINE_INTERVAL_1MINUTE)
            # open1, close = get_historic_klines(symbol, "6 months ago UTC", "now UTC", Client.KLINE_INTERVAL_1DAY)
            # open1, close = get_historic_klines(symbol, "24 hours ago UTC", "now UTC", Client.KLINE_INTERVAL_15MINUTE)
            # open1, close = get_historic_klines(symbol, "1 Dec, 2017", "1 Jan, 2018", Client.KLINE_INTERVAL_30MINUTE)
            # open1, close = get_historic_klines(symbol, "12 hours ago UTC", "Now UTC", Client.KLINE_INTERVAL_1MINUTE)

            if ((open1 / close) - 1) * 100 >= float(args.buy_limit):
                print('Buy!')

                try:
                    '''
                    order = client.create_test_order(
                        symbol='BNBBTC',
                        side=Client.SIDE_BUY,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=100)
                    '''
                    order = client.order_limit_buy(
                        quantity=100,
                        symbol='BNBBTC',
                        price='0.00001')
                except Exception as e:
                    print("\n \n \nATTENTION: NON-VALID CONNECTION WITH BINANCE \n \n \n", e)

            if ((open1 / close) - 1) * 100 == float(args.sell_limit) or (close == open1):
                print("Sell!")

                try:
                    order = client.order_limit_sell(
                        symbol='BNBBTC',
                        quantity=100,
                        price='0.00001')
                except Exception as e:
                    print("\n \n \nATTENTION: NON-VALID CONNECTION WITH BINANCE \n \n \n", e)

        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('buy_limit', default=1, help="Choose percent at which it should buy")
    parser.add_argument('sell_limit', default=0.5, help="Choose percent at which it should sell")
    args = parser.parse_args()
    buy_sell(args)
