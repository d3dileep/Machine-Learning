#   @author= Dileep
#   @Date=2018-09-05


import os
import sys

import numpy as np
import pandas as pd
import datetime
from datetime import datetime,timedelta, date

def main(args):

    #date_entry = input('Enter a date in YYYY-MM-DD format')
    #year, month, day = map(int, date_entry.split('-'))
    #date1 = datetime(year, month, day).date()
#    sys.path.insert(0, args.python_path)
    #valid_date(args.s)
    #args.date1= date1
    get_bigquery(args)
    #to_bigquery(args)

def get_bigquery(args):
    #from utils.Storage import Storage
    #storage_client = Storage(args.google_key_path)
    q1="""SELECT
  DISTINCT(constituent_name),
  constituent_id,
  opening_price
FROM
  `pecten_dataset.stockprice_historical_collection`
WHERE
  constituent_name!='DAX'
  AND constituent_name!='Wirecard AG'
  AND date='{} 00:00:00'""".format(args.date)
    q2="""SELECT
  * EXCEPT (row_number)
FROM (
  SELECT
    constituent_id,
    datetime,
    close,
    ROW_NUMBER() OVER (PARTITION BY constituent_id ORDER BY datetime DESC) row_number
  FROM
    `igenie-project.pecten_dataset.stockprice_ticker_data_collection_gs`)
WHERE
  row_number = 1 and constituent_id!='DAX'"""
    query1=pd.read_gbq(q1,project_id='igenie-project', dialect='standard')
    query2=pd.read_gbq(q2,project_id='igenie-project', dialect='standard')
    #a=['Wirecard AG']
    #b=['DAX']
    #query1.drop('Wirecard AG', axis=0,inplace=True)
    #query2.drop('DAX',axis=0,
    dfinal = query2.merge(query1, on="constituent_id", how = 'inner')
    #pd.merge(query2, query1, on="constituent_id")
    #qq={}
    to_insert=[]
    #df = pd.DataFrame(columns=['constituent_name','return'])
    for i in range(len(dfinal)):
        q1qq=(dfinal.iloc[i,2]/dfinal.iloc[i,4])-1
        qq={}
        qq["constituent_name"]=dfinal.iloc[i,3]
        qq["current_price"]= dfinal.iloc[i,2]
        qq["return"]=round(q1qq,6)
        to_insert.append(qq)
    df = pd.DataFrame(to_insert,columns=['constituent_name','current_price','return'])
    df.to_gbq('pecten_dataset_dev.ticker_return','igenie-project',if_exists='replace')
    print(to_insert,df,dfinal)

#def valid_date(s):
#    try:
#        return datetime.strptime(s, "%Y-%m-%d")
#    except ValueError:
#        msg = "Not a valid date: '{0}'.".format(s)
#        raise argparse.ArgumentTypeError(msg)
#    date1=s
#    return date1
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('python_path', help='The connection string')
    #parser.add_argument('google_key_path', help='The path of the Google key')
    #parser.add_argument('param_connection_string', help='The connection string')
    parser.add_argument("date",  help="The Start Date - format YYYY-MM-DD")
    args = parser.parse_args()
    #sys.path.insert(0, args.python_path)
    main(args)
