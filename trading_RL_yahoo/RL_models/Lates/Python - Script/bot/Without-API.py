import lxml.html as lh
import time
import urllib.request
import argparse
import urllib.request, urllib.parse, urllib.error
import datetime
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
import yfinance as yf # importing yahoo finance
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

import pandas as pd
import logging
import math
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

tz = pytz.timezone('Asia/Kolkata')

coloredlogs.install(level="DEBUG")
switch_k_backend_device()

window_size = 10
batch_size = 32
ep_count = 1
strategy = "double-dqn"
model_name = "model_double-dqn_GOOG_50_10"
pretrained = True
debug = False

path1 = os.getcwd()
path = path1 + '/chromedriver'
ignored_exceptions=(StaleElementReferenceException,)
options = webdriver.ChromeOptions()

options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--headless')
options.add_argument('log-level=0') # added additionally
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(executable_path=path , options=options)

#---------------------------------------------------------------------------------------------------------
def get_train_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="max")
    df = df[[ 'Open','Close','High','Low', 'Volume']]
    df = df[df.index > '2015-02-10']
    return df[:-200],df[-200:-100]

#--------------------------------------------------------------------------------------------------------------------
def train(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_double-dqn_GOOG_50", pretrained=True,
         debug=False):
    """ 
    Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.
    Args: [python train.py --help]
    """
    print("Started the model training for the {}".format(symbol))
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = train_stock
    val_data = val_stock
    initial_offset = np.array(val_data)[1] - np.array(val_data)[0]


    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)
    print("Training the model completed!!")

#---------------------------------------------------------------------------------------------------------------------------
def Real(url,count):
    if count == 0:        
        driver.get(url)
        #print(driver)
    else:
        driver.refresh()
        time.sleep(2)

    infile = driver.page_source
    doc = lh.fromstring(infile)
    live = doc.xpath('/html/body/div[1]/div/div/div[1]/div/div[2]/div/div/div[4]/div/div/div/div[3]/div/div/span[1]')
    live = float(live[0].text.replace(',',''))
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
    url = 'https://finance.yahoo.com/quote/{}?p={}&.tsrc=fin-srch'.format(symbol,symbol)

    while count < window_size:
        live = Real(url,count)
        print(live)
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
    step_size = 10
    #print(step_size)
    datetime_list = []
    p = []
    quantity = {}
    status= []
    profit = []
    fq = []
    time_now = datetime.datetime.now(tz).time()
    while (datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(23,49 , tzinfo=tz)):
    #while count<11:
        live = Real(url,count)
        count += 1
        time_now = datetime.datetime.now(tz).time()
        #print(live)
        data.append(live)
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        
        action = agent.act(state, is_eval=True)

        print("Live Price: ",live)
        if action == 1 and number_of_buys <= max_transaction and len(agent.inventory)<=quantity_1:
            datetime_list.append(datetime.datetime.now(tz))
            p.append(live)
            status.append("BOUGHT")
            profit.append(0)
            fp = floatPrecision((max_amount / (quantity_1 * live)),2)
            quantity[live] = fp
            fq.append(fp)
 
            agent.inventory.append(data[t+window_size])
            history.append((data[t+window_size], "BUY"))
            number_of_buys += 1
            df1 = pd.DataFrame({'Datetime': [datetime.datetime.now()], 'Symbol': [symbol], 'Buy/Sell': ['Buy'],
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
                        q = float(floatPrecision(float(quantity[i])*0.9989,2))
                        pft = temp * q 
                        agent.inventory.remove(i)
                        delta = pft
                        reward = delta #max(delta, 0)
                        total_profit += delta
                        del quantity[i]
                        datetime_list.append(datetime.datetime.now(tz))
                        p.append(live)
                        status.append("SOLD")
                        profit.append(delta)
                        fq.append(q )

                        history.append((data[t+window_size], "SELL"))
                        df2 = pd.DataFrame({'Datetime': [datetime.datetime.now()], 'Symbol': [symbol], 'Buy/Sell': ['Sell'],
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
                    
            return str(total_profit), history
#---------------------------------------------------------------------------------------------------------------------------
def main(symbol):
    price = []
    window_size = 10
    time_now = datetime.datetime.now(tz).time()
    model_name='model_double-dqn_GOOG_50_10'
    agent = Agent(window_size, pretrained=True, model_name=model_name)
    print("[INFO] Model initialised successfully[INFO]")
    profit, history = evaluate_model1(agent, symbol, price, window_size)
    print("Profit:", profit)
    '''
    with open("profit.txt", "w") as text_file:
        print("Total Profit {}".format(profit), file=text_file)
    '''

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
def run_method(val,_id):
    try:
        global symbol
        symbol = val
        _id = int(_id)
        if(_id==0):
            print("[INFO]Getting Training DATA[INFO]")
            train_stock,val_stock = get_train_data(symbol)
            print(train_stock.head(2))
            print(val_stock.head(2))
            train(train_stock, val_stock, window_size, batch_size, ep_count, strategy, model_name, pretrained)
            main(symbol)
        elif(_id==1):
        	print("[INFO]Using Pretrained Model.... Prediction strated[INFO]")
        	main(symbol)
        else:
        	print("Wrong argument")

    except KeyboardInterrupt:
        print("Aborted")
#---------------------------------------------------------------------------------------------------------------------------
run_method(sys.argv[1],sys.argv[2])
