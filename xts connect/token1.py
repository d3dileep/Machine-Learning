from Connect import XTSConnect
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
import configparser

def xts_data_token(API_KEY = '576bffb2375a5011e67154', API_SECRET = 'Mlqd885$q7'):
    XTS_API_BASE_URL = "https://xts.compositedge.com"
    source = "WEBAPI"
    xt = XTSConnect(API_KEY, API_SECRET, source= source)
    response = xt.marketdata_login()
    set_marketDataToken = response['result']['token']
    set_muserID = response['result']['userID']
    token_filename = "token_nif.ini"
    print(token_filename )
    text_file = open(token_filename, "a")
    text_file.write("[datatoken] \n token=%s \n" % set_marketDataToken)
    text_file.write("[datauser] \n user=%s \n" % set_muserID)
    text_file.close()

def xts_order_token( API_KEY = 'e3a64d975100976e6c3303', API_SECRET = 'Adoi655#hT'):
    XTS_API_BASE_URL = "https://xts.compositedge.com"
    source = "WEBAPI"
    xt = XTSConnect(API_KEY, API_SECRET, source= source)
    response = xt.interactive_login()
    set_marketDataToken = response['result']['token']
    set_muserID = response['result']['userID']
    token_filename = "token_nif.ini"
    print(token_filename )
    text_file = open(token_filename, "w")
    text_file.write("[ordertoken] \n token=%s \n" % set_marketDataToken)
    text_file.write("[orderuser] \n user=%s \n" % set_muserID)
    text_file.close()

xts_order_token(API_KEY = 'e3a64d975100976e6c3303', API_SECRET = 'Adoi655#hT')
xts_data_token(API_KEY = '576bffb2375a5011e67154', API_SECRET = 'Mlqd885$q7')