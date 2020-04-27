import lxml.html as lh
import time
import urllib.request
import argparse
import datetime
import pytz
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
import os
import coloredlogs

from docopt import docopt
from trading_bot.ops import get_state
from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)




tz = pytz.timezone('Asia/Kolkata')

path1 = os.getcwd()
path = path1 + '/chromedriver'

ignored_exceptions=(StaleElementReferenceException,)
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--headless')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
#options.add_argument(f'user-agent={userAgent}')
driver = webdriver.Chrome(executable_path=path , options=options)



#This function returns the table of the given url
def Real(url,count):
    if count == 0:
        
        driver.get(url)
        #print(driver)

    else:
        driver.refresh()
        time.sleep(15)

    infile = driver.page_source
    doc = lh.fromstring(infile)
    live = doc.xpath('/html/body/div[1]/div/div/div[1]/div/div[2]/div/div/div[4]/div/div/div/div[3]/div/div/span[1]')
    live = float(live[0].text.replace(',',''))
    return live 

def main(args):
    count = 0
    total_profit = 0
    t=0
    history = []
    reward = 0
    ticker = args.ticker + '.NS'
    price = []
    window_size =10
    time_now = datetime.datetime.now(tz).time()
    while(datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(15, 31, tzinfo=tz)):
        url = 'https://finance.yahoo.com/quote/{}?p={}&.tsrc=fin-srch'.format(ticker,ticker)
        print(count)
        live = Real(url,count)
        count+=1        
        price.append(live)
        if count < window_size:
           continue
        model_name='model_debug_50'  
        print(live)
        initial_offset = price[1] - price[0]
        state = get_state(price, 0, window_size + 1)
        next_state = get_state(price, t + 1, window_size + 1)
        agent = Agent(state_size=window_size, pretrained=True, model_name=model_name)
        agent.inventory = []
        profit = evaluate_model(agent,state,next_state, price, t, total_profit, history, reward, window_size=window_size)
        show_eval_result(model_name, profit, initial_offset)
        t+=1
        state = next_state
def evaluate_model(agent, state, next_state, data, t, total_profit, history, reward, window_size, debug=False):
  
    print(t)
        # select an action
    action = agent.act(state, is_eval=True)

        # BUY
    if action == 1:
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

#        done = (t == data_length - 1)
    agent.memory.append((state, action, reward, next_state))

    return total_profit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', help = 'ticker')
    args = parser.parse_args()

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(args)
    except KeyboardInterrupt:
        print("Aborted")    

