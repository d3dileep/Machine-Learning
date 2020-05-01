import requests
import lxml.html as lh
import pandas as pd
import time
import urllib.request
import argparse
import datetime
import pytz
import os
import Storage
import numpy as np
from selenium import webdriver
from collections import OrderedDict
from fake_useragent import UserAgent

global prev
prev = []


ua = UserAgent()
userAgent = ua.random
tz = pytz.timezone('Asia/Kolkata')

path1 = os.getcwd()
path = path1 + '/chromedriver'


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

    else:
        driver.refresh()
        time.sleep(15)

    infile = driver.page_source
    doc = lh.fromstring(infile)
    tr_elements = doc.xpath('//tr')
    tr_elements = tr_elements[1:502]

    col=[]
    i=0#For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content().strip()
        col.append((name,[]))


    for j in range(1,len(tr_elements)):
        #T is our j'th row
        T=tr_elements[j]
        
        #If row is not of size 500, the //tr data is not from our table 
        if len(T)!=8:
            break
        
        #i is the index of our column
        i=0
        
        #Iterate through each element of the row
        for t in T.iterchildren():
            data=t.text_content() 
            #Check if row is empty
            if i>0:
            #Convert any numerical value to integers
                try:
                    data=int(data)
                except:
                    pass
            #Append the data to the empty list of the i'th column
            col[i][1].append(data)
            #Increment i for the next column
            i+=1

    #print([len(C) for (title,C) in col])
    Dict={title:column for (title,column) in col}
    df=pd.DataFrame(Dict)
    df['LTP'] = df['LTP'].apply(lambda x: x.replace(',',''))
    df['LTP'] = df['LTP'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
    df['Volume'] = df['Volume'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
    df.rename(columns = {'%Change':'per_change'}, inplace = True)
    df.rename(columns = {'Company':'company_name'}, inplace = True)
    df = df.drop(['Buy Price', 'Sell Price', 'Buy Qty', 'Sell Qty'],axis= 1)
    df.dropna(inplace=True)
    #print(count)

    return df

def main(args):
    count = 0
    args.storage = Storage.Storage(google_key_path=args.google_key_path)
    url = 'https://www.moneycontrol.com/markets/indian-indices/top-nse-500-companies-list/7?classic=true'
    #datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(15, 31, tzinfo=tz)
    time_now = datetime.datetime.now(tz).time()
    while(datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(15, 31, tzinfo=tz)):
        try:
            df = Real(url,count)
            df['LTP'] = df['LTP'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
            ltp = list(map(float,list(df['LTP'])))
            volume = list(map(int,list(df['Volume'])))
            print(time_now)           
            df['per_change'] = df['per_change'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
            per = list(map(float,list(df['per_change'])))

            if (count == 0 ):
                df['open'] = df['LTP']
                df['close'] = df['LTP']
                df['high'] = df['LTP']
                df['low'] = df['LTP']
                df = df.drop(['LTP'],axis= 1)

                df.to_csv('Data.csv',index=False)
                prev = list(map(float,list(df['close'])))
                prev1 = list(map(float,list(df['close'])))

            elif(count == 5):
                if(prev == prev1):
                    print('Market Closed!')
                    break
                else:
                    pass
            
            else:
                df1 = pd.read_csv('Data.csv')
                high = list(map(float,list(df1['high'])))
                low = list(map(float,list(df1['low'])))


                #Since len of high,low,open,close is the same
                for i in range(len(high)):
                    if(ltp[i]>high[i]):
                        high[i] = ltp[i]

                    if(ltp[i]<low[i]):
                        low[i] = ltp[i]

                
                df1['high'] = high
                df1['low'] = low
                df1['close'] = ltp
                df1['volume'] = volume
                df1['per_change'] = per
                df1['open'] = prev
                df1["datetime"] = (datetime.datetime.now(tz))
                df1['datetime'] = df1['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                df1['datetime'] = df1['datetime'].apply(lambda x : pd.to_datetime(x))
                df1['datetime'] = df1['datetime'].dt.round('min')
                df1 = df1[['datetime','company_name', 'open', 'low', 'high', 'close', 'volume' ,'per_change']]

                df1.to_csv('Data.csv',index=False)
                prev = list(map(float,list(df1['close'])))


                records = df1.to_dict('records')
                args.storage.insert_bigquery_data(args.environment, args.table, records)

            count+=1
            t = 50
            while t>=0:
                mins,secs = (00,t)
                timer = '{:02d}:{:02d}'.format(mins,secs)
#                print(timer, end='\r')
                time.sleep(1)
                t -= 1
            
            time_now = datetime.datetime.now(tz).time()



        except KeyboardInterrupt:
            break





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.environment = 'development'
    args.table = "mc_realtime_quote"

    main(args)
