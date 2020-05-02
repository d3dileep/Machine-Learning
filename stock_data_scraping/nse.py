import lxml.html as lh
import pandas as pd
import time
import urllib.request
import argparse
import datetime
import pytz
import os
import Storage
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException

global prev
prev = []


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

    
    
    

    company = []
    ltp = []
    per = []


    infile = driver.page_source
    doc = lh.fromstring(infile)
    

    for i in range(1, 37):



        name = doc.xpath('/html/body/section/section[4]/div[7]/div[3]/div[2]/div[{}]/ul/li[1]/a'.format(i))
        company.append(name[0].text)

        live = doc.xpath('/html/body/section/section[4]/div[7]/div[3]/div[2]/div[{}]/ul/li[2]/span'.format(i))
        live = float(live[0].text.replace(',',''))
        ltp.append(live)
        
        perchg = doc.xpath('/html/body/section/section[4]/div[7]/div[3]/div[2]/div[{}]/ul/li[5]/span'.format(i))
        perchg = float(perchg[0].text.replace(',',''))
        per.append(perchg)

        
    


    
    df = pd.DataFrame({'company_name':company,'LTP': ltp,'per_change':per})

    
    return df

def main(args):
    count = 0
    args.storage = Storage.Storage(google_key_path=args.google_key_path)
    url = 'https://economictimes.indiatimes.com/marketstats/pid-40,exchange-nse,sortby-value,sortorder-desc.cms'
    #datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(15, 31, tzinfo=tz)
    time_now = datetime.datetime.now(tz).time()
    while(datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(15, 31, tzinfo=tz)):
        try:
            df = Real(url,count)
            #df['LTP'] = df['LTP'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
            ltp = list(map(float,list(df['LTP'])))
            print(time_now)            

            #df['per_change'] = df['per_change'].apply(lambda x: pd.to_numeric(x,errors='coerce'))
            per = list(map(float,list(df['per_change'])))

            if (count == 0 ):
                df['open'] = df['LTP']
                df['close'] = df['LTP']
                df['high'] = df['LTP']
                df['low'] = df['LTP']
                df = df.drop(['LTP'],axis= 1)

                df.to_csv('nse.csv',index=False)
                prev = list(map(float,list(df['close'])))
                prev1 = list(map(float,list(df['close'])))

            elif(count == 5):
                if(prev == prev1):
                    print('Market Closed!')
                    break
                else:
                    pass
            
            else:
                df1 = pd.read_csv('nse.csv')
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
                df1['per_change'] = per
                df1['open'] = prev
                df1["datetime"] = (datetime.datetime.now(tz))
                df1['datetime'] = df1['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
                df1['datetime'] = df1['datetime'].apply(lambda x : pd.to_datetime(x))
                df1['datetime'] = df1['datetime'].dt.round('min')
                df1 = df1[['datetime','company_name', 'open', 'low', 'high', 'close', 'per_change']]

                df1.to_csv('nse.csv',index=False)
                prev = list(map(float,list(df1['close'])))


                records = df1.to_dict('records')
                args.storage.insert_bigquery_data(args.environment, args.table, records)

                       
            count+=1
            t = 50
            while t>=0:
                mins,secs = (00,t)
                timer = '{:02d}:{:02d}'.format(mins,secs)
                #print(timer, end='\r')
                time.sleep(1)
                t -= 1

            time_now = datetime.datetime.now(tz).time()

        except KeyboardInterrupt:
            break
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.environment = 'development'
    args.table = "nifty_sector"

    main(args)          
