import logging
import coloredlogs
import lxml.html as lh
import time
import urllib.request
import argparse
import urllib.request, urllib.parse, urllib.error
import datetime
import pytz
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
import os
import coloredlogs
import json
import ssl
from docopt import docopt
from trading_bot.ops import get_state
from trading_bot.agent import Agent
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)
from binance.client import Client

tz = pytz.timezone('Asia/Kolkata')
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
api_key = "JB80jU6aTMYXcLapXKPc2Rmnn12CUcm3l7RlnYAT7w8esCFAKYOTmd4bAMWxB33U"
api_secret = "FsbKTSb7lCGLpQWoBad9Jobe8xpi177c2KrQ6Q31e86dUA5WgUqaliqHsILk7n5s"
client = Client(api_key, api_secret)
client.get_deposit_address(asset='USDT')

global symbol 
symbol = 'BTCUSDT'

def Real(client):
    live = float(client.get_symbol_ticker(symbol = symbol)['price']);

    return live
    
def evaluate_model(agent, data, window_size, debug):
    count = 0
    while count < window_size:
        live = Real(client)
        data.append(live)
        count += 1
    total_profit = 0
    history = []
    agent.inventory = []
    state = get_state(data, 0, window_size + 1)
    for t in range(100):   
        live = Real(client)
        data.append(live)
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1 and t < 88 and len(agent.inventory)< 10:
            agent.inventory.append(data[t])
            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == (100 - 1))
        agent.memory.append((state, action, reward, next_state, done))
#        print(agent.inventory)
        state = next_state
        time.sleep(10)
        if done:
            return total_profit, history
            
def main(args):
    price = []
    window_size =10
    time_now = datetime.datetime.now(tz).time()
    model_name='model_double-dqn_GOOG_50_30'

    agent = Agent(window_size, pretrained=True, model_name=model_name)
    profit, history = evaluate_model(agent, price, window_size, debug=True)
    print("Profit:", profit)
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
    
coloredlogs.install(level="DEBUG")
switch_k_backend_device()
args = argparse.Namespace(ticker=symbol)
try:
    main(args)
except KeyboardInterrupt:
    print("Aborted")  
