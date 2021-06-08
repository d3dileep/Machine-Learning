from Connect2 import XTSConnect
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

class XTS_parse:
    def __init__(self, token, userID, isInvestorClient):
        self.XTS_API_BASE_URL = "https://xts.compositedge.com"
        self.source = "WEBAPI"

        self.xt = XTSConnect(source=self.source, token=token, userID=userID, isInvestorClient=isInvestorClient)
        
        self.response = self.xt.get_config()
        
    def get_id(self,item):
        if item=='NIFTY':
            self.quantity = 75
        if item=='BANKNIFTY':
            self.quantity = 25      
        self.data = pd.DataFrame(self.xt.search_by_scriptname(searchString=item)['result'])[['ExchangeSegment', 'ExchangeInstrumentID','Description']]
        return self.quantity,self.data

    def get_latest_price(self,item):
        id1 = self.data.loc[self.data['Description'] == item, ['ExchangeSegment','ExchangeInstrumentID']].iloc[0,1]
        instruments = [{'exchangeSegment': 2, 'exchangeInstrumentID': int(id1)}]
        jsoned = json.loads(self.xt.get_quote(Instruments=instruments,xtsMessageCode=1502,  publishFormat='JSON')['result']['listQuotes'][0])
        #print(jsoned)
        dt =float(jsoned['Touchline']['LastTradedPrice'])
        return dt

    def get_quote_oi(self, item):
        """Get Quote Request"""
        id1 = self.data.loc[self.data['Description'] == item, ['ExchangeSegment','ExchangeInstrumentID']].iloc[0,1]
        instruments = [{'exchangeSegment': 2, 'exchangeInstrumentID': int(id1)}]
        response = self.xt.get_quote( Instruments=instruments, xtsMessageCode=1510, publishFormat='JSON')
        return int(json.loads(response['result']['listQuotes'][0])['OpenInterest'])

    def read_data(self, item, interval, exchange,division='NIFTY', days=False, prime=False):
        if item == 'NIFTY':
            symbol_1=26000
        if item == 'BANKNIFTY':
            symbol_1=26001
        elif item not in ['NIFTY','BANKNIFTY']:
            symbol_1 = self.data.loc[self.data['Description'] == item, ['ExchangeSegment','ExchangeInstrumentID']].iloc[0,1]
        #print(symbol_1)
        tzinfo = pytz.timezone('Asia/Kolkata')
        #print(item)
        now = datetime.now(tz= tzinfo)#tz= tzinf
        today = now.date()
        current_time = time(now.hour, now.minute, now.second)
      # print(current_time,today, today.weekday())

        to_d = datetime.combine(today, current_time)

        if prime:
            from_d = datetime.combine(today -timedelta(days=6), time(9,15,00) )
        if days:
            from_d = datetime.combine(days, time(9,15,00) )

        from_d = from_d.strftime("%b %d %Y %H%M%S")
        to_d = to_d.strftime("%b %d %Y %H%M%S")

        try:
            data = self.xt.get_ohlc(exchangeSegment=exchange,exchangeInstrumentID=symbol_1,
                startTime=from_d,endTime=to_d,compressionValue=interval)['result']['dataReponse']
            result = [x.strip() for x in data.split(',')]
            value =[]
            for item in result:
              data = [x.strip() for x in item.split('|')]
              value.append(data)
            data = pd.DataFrame(value,columns=['date','open','high','low','close','volume','oi','red'])[['date','open','high','low','close','oi']]
            data['date'] = pd.to_datetime(data['date'], unit='s')
            
            
            data[['open','high','low','close','oi']] = data[['open','high','low','close','oi']].apply(pd.to_numeric, errors='ignore')

        except Exception as e:
            print(item, e)
            pass
        return data, to_d

    def get_instr_list(self):
        nsefo_instr_url = 'http://public.fyers.in/sym_details/NSE_FO.csv'
        s=requests.get(nsefo_instr_url).content
        fo_instr=pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)
        fo_instr[1] = fo_instr[1].apply(lambda x: x.upper())

        return fo_instr

    def get_options_contract(self, underlying, opt_type, strike, nearest_expiry, monthend_expiry):

        if monthend_expiry == 'YES':
            fyers_symbol = 'NSE:' + underlying + str(nearest_expiry.year - 2000) + nearest_expiry.strftime('%b').upper() + str(strike)
        else:
            fyers_symbol = 'NSE:' + underlying + str(nearest_expiry.year - 2000) + str(int(nearest_expiry.strftime('%m'))) + nearest_expiry.strftime('%d') + str(strike)
        td_symbol = underlying + str(nearest_expiry.year - 2000) + nearest_expiry.strftime('%m') + nearest_expiry.strftime('%d') + str(strike)

        if opt_type == 'CE':
            fyers_symbol = fyers_symbol + 'CE'
            td_symbol = td_symbol + 'CE'
        else:
            fyers_symbol = fyers_symbol + 'PE'
            td_symbol = td_symbol + 'PE'

        return fyers_symbol.replace("NSE:",''), td_symbol

    def roundup(self,x,y):
        return int(math.ceil(x / y)) * y
    def rounddown(self,x,y):
        return int(math.floor(x / y)) * y
    def roundoff(self,x):
        return round(x,2)

class XTS_order:
  def __init__(self, token, userID, isInvestorClient):
    self.XTS_API_BASE_URL = "https://xts.compositedge.com"
    self.source = "WEBAPI"

    self.xt = XTSConnect(self.source, token=token, userID=userID, isInvestorClient=isInvestorClient)
    
  def place_order(self, data, quantity, item, action, q_multiplier):
    id1 = int(data.loc[data['Description'] == item, ['ExchangeSegment','ExchangeInstrumentID']].iloc[0,1])
    order = self.xt.place_order(exchangeSegment='NSEFO', exchangeInstrumentID=id1, productType='NRML', orderType='MARKET',
            orderSide=action, timeInForce=self.xt.VALIDITY_DAY, orderQuantity= quantity * q_multiplier, orderUniqueIdentifier="454845", disclosedQuantity=0, limitPrice=0, stopPrice=0)

  def exit_order(self, data, quantity, item, q_multiplier):
    id1 = int(data.loc[data['Description'] == item, ['ExchangeSegment','ExchangeInstrumentID']].iloc[0,1])
    exit = self.xt.squareoff_position(exchangeSegment='NSEFO', exchangeInstrumentID=id1, productType='NRML',squareoffMode='Netwise', positionSquareOffQuantityType='ExactQty', squareOffQtyValue=quantity * q_multiplier,blockOrderSending=True, cancelOrders=True)

  def get_positions(self):
    resp = self.xt.get_position_netwise()
    return resp

  def get_balance(self):
    """Get Balance API call grouped under this category information related to limits on equities, derivative,
    upfront margin, available exposure and other RMS related balances available to the user."""
    if self.xt.isInvestorClient:
      try:
        params = {}
        if not self.xt.isInvestorClient:
            params['clientID'] = self.xt.userID
        response = self.xt._get('user.balance', params)
        return response
      except Exception as e:
        return response['description']
    else:
      print("Balance : Balance API available for retail API users only, dealers can watch the same on dealer "
              "terminal")
  
  
  def get_holding(self):
    """Holdings API call enable users to check their long term holdings with the broker."""
    try:
      params = {}
      if not self.xt.isInvestorClient:
        params['clientID'] = self.xt.userID

      response = self.xt._get('portfolio.holdings', params)
      return response
    except Exception as e:
      return response['description']            
                  
