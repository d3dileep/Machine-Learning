import lxml.html as lh
import time
import urllib.request
import argparse
import urllib.request, urllib.parse, urllib.error
from datetime import datetime
import dateparser
import argparse
import pytz
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
import os
import sys
import coloredlogs
import json
import ssl
from docopt import docopt
from trading_bot.ops import get_state
from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    show_train_result,
    switch_k_backend_device
)

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import logging
import math
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------------------
def get_historic_klines(symbol, start, end, interval):
    klines = get_historical_klines(symbol, interval, start, end)
    #print(klines[-1])
    ochl = []
    for kline in klines:
        time1 = int(kline[0])
        open1 = float(kline[1])
        high = float(kline[2])
        low = float(kline[3])
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
    return ochl
#---------------------------------------------------------------------------------------------------------
def make_csv_file(symbol):
    ochl = get_historic_klines(symbol, "5 days ago UTC", "now UTC", Client.KLINE_INTERVAL_5MINUTE)
    df = pd.DataFrame(ochl, columns=["Date", "Open", "Close", "High", "Low", "Volume"])                
    f_name = str(symbol)
    for i in df.index:
        df.at[i,'Date'] = datetime.fromtimestamp((int(df.at[i,'Date']))/1000.0).date()
    df.iloc[1:1001].to_csv(f_name+"_TRAINING.csv", columns=["Date", "Open", "Close", "High", "Low", "Volume"])
    print(f_name+"_TRAINING.csv created!")
    df.iloc[1001:1440].to_csv(f_name+"_TESTING.csv", columns=["Date", "Open", "Close", "High", "Low", "Volume"])
    print(f_name+"_TESTING.csv created!")
#--------------------------------------------------------------------------------------------------------------------
def train(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_double-dqn_GOOG_50", pretrained=True,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.
    Args: [python train.py --help]
    """
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)
    print("Training the model completed!!")
#---------------------------------------------------------------------------------------------------------------------------
def Real(client, symbol):
    live = float(client.get_symbol_ticker(symbol=symbol)['price']);
    time.sleep(1)
    return live
#----------------------------------------------------------------------------------------------------------------
def floatPrecision(f, n):
    n = int(math.log10(1 / float(n)))
    f = math.floor(float(f) * 10 ** n) / 10 ** n
    f = "{:0.0{}f}".format(float(f), n)
    return str(int(f)) if int(n) == 0 else f
#-----------------------------------------------------------------------------------------------------------------
def evaluate_model1(agent, symbol, data, window_size, debug=False):
    count = 0
    while count < window_size:
        live = Real(client, symbol)
        data.append(live)
        count += 1
    total_profit = 0
    history = []
    agent.inventory = []
    state = get_state(data, 0, window_size + 1)
    number_of_buys = 0
    max_transaction = 100  #  maximum buy/sell limit
    quantity_1 = 5         #  divide max amount in the number of parts
    
    max_amount = 1000      #  maximum  amount bot is allowed to trade with
    max_loss = -5          #  stop loss amount in dollar
    t = 0
    step_size = float(next(filter(lambda f: f['filterType'] == 'LOT_SIZE', client.get_symbol_info(symbol)['filters']))['stepSize'])
    #print(step_size)
    datetime_list = []
    p = []
    quantity = {}
    status= []
    profit = []
    fq = []
    
    while True:
        live = Real(client, symbol)
        print(live)
        data.append(live)
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        action = agent.act(state, is_eval=True)

        #print("Live Price: ",live)
        if action == 1 and number_of_buys <= max_transaction and len(agent.inventory)<=quantity_1:
            datetime_list.append(datetime.now(tz))
            p.append(live)
            status.append("BOUGHT")
            profit.append(0)
            fp = floatPrecision((max_amount / (quantity_1 * live)),step_size )
            quantity[live] = fp
            fq.append(fp)
 
            agent.inventory.append(data[t+window_size])
            history.append((data[t+window_size], "BUY"))
            number_of_buys += 1
            df1 = pd.DataFrame({'Datetime': [datetime.now()], 'Symbol': [symbol], 'Buy/Sell': ['Buy'],
                                                'Quantity': [fp], 'Price': [live], 'Profit/loss': [0]})
            df1['Datetime'] = df1['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            if not os.path.isfile('{}.csv'.format(symbol)):
                df1.to_csv('{}.csv'.format(symbol), index=False)
            else:
                df1.to_csv('{}.csv'.format(symbol), index=False, mode='a', header=False)            
            print("Buy at: {}".format(format_currency(data[t+window_size])))
        
        elif action == 2 and len(agent.inventory) > 0:
            if agent.inventory != []:
                for i in agent.inventory:
                    temp = data[t+window_size] - i
                    if temp > 0:
                        q = float(floatPrecision(float(quantity[i])*0.9989,step_size))
                        pft = temp * q 
                        agent.inventory.remove(i)
                        delta = pft
                        reward = delta #max(delta, 0)
                        total_profit += delta
                        del quantity[i]
                        datetime_list.append(datetime.now(tz))
                        p.append(live)
                        status.append("SOLD")
                        profit.append(delta)
                        fq.append(q )

                        history.append((data[t+window_size], "SELL"))
                        df2 = pd.DataFrame({'Datetime': [datetime.now()], 'Symbol': [symbol], 'Buy/Sell': ['Sell'],
                                            'Quantity': [q], 'Price': [live], 'Profit/loss': [pft]})
                        df2['Datetime'] = df2['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                        df2.to_csv('{}.csv'.format(symbol), index=False, mode='a', header=False)
                        print("Sell at: {} | Position: {}".format(
                                format_currency(data[t+window_size]), format_position(delta)))
                        
        
        else:
            history.append((data[t], "HOLD"))
            if False:
                logging.debug("Hold at: {}".format(format_currency(data[t+window_size])))
        time.sleep(10)
        done=False
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        t += 1
        if sum(profit) <= max_loss:
            break
        if agent.inventory == [] and number_of_buys > max_transaction:
                    
            return total_profit, history
#---------------------------------------------------------------------------------------------------------------------------
def main(symbol):
    price = []
    window_size = 10
    time_now = datetime.now(tz).time()
    model_name='model_double-dqn_GOOG_50_10'

    agent = Agent(window_size, pretrained=True, model_name=model_name)
    profit, history = evaluate_model1(agent, symbol, price, window_size)
    print("Profit:", profit)
    '''
    buys = sells = holds = 0
    for i in history:
        if i[1] == "BUY":
            buys += 1
        elif i[1] == "SELL":
            sells += 1
        elif i[1] == "HOLD":
            holds += 1
    print("BUYS Percentage:", (buys/len(history)) * 100)
    print("SELLS Percentage:", (sells/len(history)) * 100)
    print("HOLDS Percentage:", (holds/len(history)) * 100)
    '''
#---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()
    args = argparse.Namespace(ticker='BTCUSDT')
    symbol = sys.argv[1]
    train_stock = symbol+"_TRAINING.csv"
    val_stock = symbol+"_TESTING.csv"
    window_size = 10
    batch_size = 32
    ep_count = 10
    strategy = "double-dqn"
    model_name = "model_double-dqn_GOOG_50_10"
    pretrained = True
    debug = False
    tz = pytz.timezone('Asia/Kolkata')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    api_key = "JB80jU6aTMYXcLapXKPc2Rmnn12CUcm3l7RlnYAT7w8esCFAKYOTmd4bAMWxB33U"
    api_secret = "FsbKTSb7lCGLpQWoBad9Jobe8xpi177c2KrQ6Q31e86dUA5WgUqaliqHsILk7n5s"
    client = Client(api_key, api_secret)
    client.get_deposit_address(asset='USDT') # select the asset with which you want to pay as base coiny
    try:
        make_csv_file(symbol)
        train(train_stock, val_stock, window_size, batch_size, ep_count, strategy, model_name, pretrained)
        os.remove("{}_TRAINING.csv".format(symbol))
        os.remove("{}_TESTING.csv".format(symbol))
        main(symbol)
    except KeyboardInterrupt:
        print("Aborted")
#---------------------------------------------------------------------------------------------------------------------------

