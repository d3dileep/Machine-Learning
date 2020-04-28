import requests
import lxml.html as lh
import pandas as pd
import time
import numpy as np
import urllib.request
import json
from selenium import webdriver
import datetime
import os
import pytz

tz = pytz.timezone('Asia/Kolkata')


url = 'https://www.moneycontrol.com/markets/indian-indices/top-nse-500-companies-list/7?classic=true'
count = 0
global prev
prev = [0 for i in range(500)]
global old
old = [0 for i in range(500)]

path1 = os.getcwd()
path = path1 + '/chromedriver'


options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--headless')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
#options.add_argument(f'user-agent={userAgent}')
driver = webdriver.Chrome(executable_path=path , options=options)




# Alert Function, It is called every 30 seconds.

def change(present):

    time_now = datetime.datetime.now(tz).time()
    data = pd.read_csv('alert.csv')
    data1 = data.copy()

    if(time_now<datetime.time(10, 00, tzinfo=tz) == True):
      #Before 10:00 
      percent = [x - y  for x, y in zip(present, prev) ]

      #data['Company'] = df['Company']
      #data['LTP'] = list(map(float,list(data['LTP'].apply(lambda x: x.replace(',','')))))
      data['%Change'] = percent
      data['Previous Value'] = prev
      data['Present Value'] = present
      #data['Volume'] = df['Volume']

      #data1['Company'] = df['Company']
      #data1['LTP'] = list(map(float,list(data1['LTP'].apply(lambda x: x.replace(',','')))))
      data1['%Change'] = percent
      data1['Previous Value'] = prev
      data1['Present Value'] = present
      #data1['Volume'] = df['Volume']

      data = data[(data['%Change'] >= 1.0) & (data['LTP'] > 10.00)]
      data1 = data1[(data1['%Change'] <= -1.0) & (data1['LTP'] > 10.00)]

    else:  
      percent = [x - y for x, y in zip(present, prev) ]
      #print(percent)
      data = pd.read_csv('alert.csv')
      data1 = data.copy()

      #data['Company'] = df['Company']
      #data['LTP'] = list(map(float,list(data['LTP'].apply(lambda x: x.replace(',','')))))
      data['%Change'] = percent
      data['Previous Value'] = prev
      data['Present Value'] = present
      #data['Volume'] = df['Volume']

      #data1['Company'] = df['Company']
      #data1['LTP'] = list(map(float,list(data1['LTP'].apply(lambda x: x.replace(',','')))))
      data1['%Change'] = percent
      data1['Previous Value'] = prev
      data1['Present Value'] = present
      #data1['Volume'] = df['Volume']

      data = data[(data['%Change'] >= 1.0) & (data['Volume'] > 100000) & (data['LTP'] > 10.00)]
      data1 = data1[(data1['%Change'] <= -1.0) & (data1['Volume'] > 100000) & (data1['LTP'] > 10.00)]

    data.sort_values("%Change", axis = 0, ascending = True, inplace = True, na_position ='last')
    data1.sort_values("%Change", axis = 0, ascending = True, inplace = True, na_position ='last') 
    
    return data,data1

#This function returns the table of the given url

def Real(url,count):

    if count == 0:
            
        driver.get(url)

    else:
        driver.refresh()
        time.sleep(3)

    infile = driver.page_source
    doc = lh.fromstring(infile)
    tr_elements = doc.xpath('//tr')
    tr_elements = tr_elements[1:502]


    #print([len(T) for T in tr_elements[:]])
    col=[]
    i=0#For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content().strip()
        #print('%d:"%s"'%(i,name))
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
    df = df.drop(['Buy Price', 'Sell Price', 'Buy Qty', 'Sell Qty'],axis= 1)
    df.dropna(inplace=True)
    return df

time_now = datetime.datetime.now(tz).time()
while(datetime.time(9, 14, tzinfo=tz) < time_now < datetime.time(16, 1, tzinfo=tz)):

  try:
    df = Real(url,count)
    df.to_csv('alert.csv',index=False)
    
    count+=1

    if(count !=1): #1st loop is ignored
        Result,Result1 = change(list(map(float,list(df['%Change']))))

        if(not Result.empty and count !=1):
            #print('#'*50)
            print("Alert! Increased")
            #print(' ')
            print(Result)
            #print('#'*50)
            #print(' ')
            Result.to_json('Increase_alert.json', orient='records')


        if(not Result1.empty and count !=1):
            #print('#'*50)
            print("Alert! Decreasing")
            #print(' ')
            print(Result1)
            #print('#'*50)
            #print(' ')
            Result1.to_json('Decrease_alert.json', orient='records')
        
        if(Result.empty and count !=1):
            a= [{}]
            with open('Increase_alert.json', 'w') as json_file:
                json.dump(a, json_file)
        
        if(Result1.empty and count !=1):
            a= [{}]
            with open('Decrease_alert.json', 'w') as json_file:
                json.dump(a, json_file)

    
        time_now = datetime.datetime.now(tz).time()

         



    prev = df['%Change']
    #By default the values are turned into str, so to convert to float we use the below statement
    prev = list(map(float,list(prev)))
    #print(prev)

    #Count Down for 60 seconds
    t = 60
    while t>=0:
        mins,secs = (00,t)
        timer = '{:02d}:{:02d}'.format(mins,secs)
        #print(timer)
        time.sleep(1)
        t -= 1
    
    time_now = datetime.datetime.now(tz).time()

  except KeyboardInterrupt:
      break

if(time_now>=datetime.time(16, 1, tzinfo=tz) == True):
    print('Market Closed')
    a= [{}]
    with open('Increase_alert.json', 'w') as json_file:
        json.dump(a, json_file)

    with open('Decrease_alert.json', 'w') as json_file:
        json.dump(a, json_file)
#Press stop button to stop the loop
#Look at try.csv for nify500 data 
