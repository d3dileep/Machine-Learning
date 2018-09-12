import os
import sys
from fbprophet import Prophet
import numpy as np
import pandas as pd
import datetime
from datetime import datetime,timedelta, date

def main(args):
    sys.path.insert(0, args.python_path)
    get_bigquery(args)
    predict(args)

def remove_weekends( dataframe):
        
        # Reset index to use ix
    dataframe = dataframe.reset_index(drop=True)
        
    weekends = []
        
        # Find all of the weekends
    for i, date in enumerate(dataframe['ds']):
        if (date.weekday()) == 5 | (date.weekday() == 6):
            weekends.append(i)
    print(weekends)        
        # Drop the weekends
    dataframe = dataframe.drop(weekends, axis=0)
        
    return dataframe

def create_model():
    model = Prophet( weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.2, changepoint_range=0.9) #instantiate Prophet
# Add monthly seasonality
    model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 8, mode='additive')
    model.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
    return model


def get_bigquery(args):
    from utils.Storage import Storage
    storage_client = Storage(args.google_key_path)
    q1="""SELECT
  date AS ds,
  closing_price AS y
FROM
`pecten_dataset.stockprice_historical_collection`
where 
constituent_name = 'DAIMLER AG'
order by 
date"""
    
    query1=pd.read_gbq(q1,project_id='igenie-project', dialect='standard')
#    print(query1)
    args.query1=query1
#    print(args.query1)

def predict(args):
    model=create_model()
    model.fit(args.query1)
    future_data = model.make_future_dataframe(periods=5, freq='D')
    forecast_data = model.predict(future_data)
    forecast_data = remove_weekends(forecast_data)
    trimmed_forecast = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    joined = forecast_data.set_index('ds')[['yhat','yhat_lower','yhat_upper']].join(args.query1.set_index('ds').y).reset_index()
    joined.columns=['Date','Predicted','Lower_confidence','Upper_confidence','Actual']
    joined.to_gbq('pecten_dataset_dev.predicted_daily_price','igenie-project',if_exists='replace')
    print(df)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('python_path', help='The connection string')
    parser.add_argument('google_key_path', help='The path of the Google key')
    #parser.add_argument('param_connection_string', help='The connection string')
    #parser.add_argument("date",  help="The Start Date - format YYYY-MM-DD")
    args = parser.parse_args()
    sys.path.insert(0, args.python_path)
    main(args)
