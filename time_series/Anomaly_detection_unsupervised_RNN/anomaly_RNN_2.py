#@author= Dileep


import os
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM,GRU
from keras.layers import  Dropout
import datetime
from datetime import datetime,timedelta, date

def main(args):
#    sys.path.insert(0, args.python_path)
    args.bound_tuner=0.035
    args.steps_to_predict=5
    get_bigquery(args)
#    create_model(args)
    make_future_dataframe(args)
    create_model(args)
    get_bigquery_compare(args)
    anomaly_check(args)
def get_bigquery(args):
#    from utils.Storage import Storage
#    storage_client = Storage(args.google_key_path)
    q1="""SELECT
  date,opening_price,daily_high,daily_low,
  closing_price
FROM
`pecten_dataset.stockprice_historical_collection`
where 
constituent_name = 'ADIDAS AG'
and date>='2013-06-01 00:00:00'
and date<='{}'
order by 
date""".format(args.min_date)

    query1=pd.read_gbq(q1,project_id='igenie-project', dialect='standard',private_key='/home/dileep/igenie-project-key.json')
#    print(query1)
    args.query1=query1
    print(args.query1.tail(5))

def get_bigquery_compare(args):
 #   from utils.Storage import Storage
  #  storage_client = Storage(args.google_key_path)
    q2="""SELECT
  date,opening_price,daily_high,daily_low,
  closing_price
FROM
`pecten_dataset.stockprice_historical_collection`
where 
constituent_name = 'ADIDAS AG'
and date>'{}'
and date<='{}'

order by 
date""".format(args.min_date, args.max_date)

    query2=pd.read_gbq(q2,project_id='igenie-project', dialect='standard',private_key='/home/dileep/igenie-project-key.json')
#    print(query1)
    args.query2=query2
    print(args.query2.tail())

def make_future_dataframe(args, include_history=True):
    """Simulate the trend using the extrapolated generative model.

        Parameters
        ----------
        periods: Int number of periods to forecast forward.
        freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
        include_history: Boolean to include the historical dates in the data
            frame for predictions.

        Returns
        -------
        pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods       
    """
    df=args.query1
    history_dates = pd.to_datetime(df['date']).sort_values()
    if history_dates is None:
        raise Exception('Model must be fit before this can be used.')
    last_date = history_dates.max()
    periods=args.steps_to_predict
    dates = pd.bdate_range(
            start=last_date,
            periods=periods + 1,  # An extra in case we include start
            freq='B')
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods

    #if include_history:
    #    dates = np.concatenate((np.array(history_dates), dates))
    date=[]
    l=pd.to_datetime(dates)
    ll=l.to_pydatetime()
    for i in ll:
        date.append(i)


#    for i in 
    extracted=df.iloc[-24:,0].tolist()
#    args.date=np.concatenate((np.array(pd.to_datetime(df.loc[-20:,'date']).sort_values()),np.array(date)))
    args.date=pd.DataFrame({'Date':extracted+date})
    args.latest_date=max(args.date['Date'])

def lstm(cols):
     unit=80
     batch_size=50
     fit1 = Sequential ()
     fit1.add (LSTM (  units=unit , activation = 'tanh', inner_activation = 'hard_sigmoid' ,return_sequences=True, input_shape =(len(cols), 1) ))
     fit1.add(Dropout(0.1))
     fit1.add(LSTM(units=unit))
     fit1.add(Dropout(0.1))
     fit1.add (Dense (units =1, activation = 'linear'))
     fit1.compile (loss ="mean_squared_error" , optimizer = "adam")
     return fit1

def create_model(args):
#    from utils.Storage import Storage
#    storage_client = Storage(args.google_key_path)
    df=args.query1
    data_to_use= df.shape[0]
    train_end =int(0.97*data_to_use)
    total_data=len(df)
    #currently doing prediction only for 3 steps ahead
    bound_tuner=args.bound_tuner
    steps_to_predict = args.steps_to_predict
    train_mse=[]
    test_mse=[]
#    forecast=[]
    
    
    w, h = df.shape[1]-1, args.steps_to_predict;
    forecast =pd.DataFrame( [[0 for x in range(w)] for y in range(h)], columns = ["P_open", "P_high", "P_low", "P_close"] )
    upper =pd.DataFrame( [[0 for x in range(w)] for y in range(h)], columns = ["U_open", "U_high", "U_low", "U_close"])
    lower =pd.DataFrame( [[0 for x in range(w)] for y in range(h)], columns = ["L_open", "L_high", "L_low", "L_close"])
    train_mse = [[0 for x in range(w)] for y in range(h)]
    test_mse = [[0 for x in range(w)] for y in range(h)]
    print(forecast)







    for k in range(1,df.shape[1]):
        print(k)
 
        yt4 = df.iloc [: ,4]    #Close price
        yt1 = df.iloc [: ,1]   #Open
        yt2 = df.iloc [: ,2]   #High
        yt3 = df.iloc [: ,3]   #Low
        args.yt=yt4 
        args.yt1=yt1
        args.yt2=yt2
        args.yt3=yt3
    #copy for loop below it and modify wht
        for i in range(steps_to_predict):
            print(i) 
            batch_size=50     
         
 
            yt_ = df.iloc[:,k].shift (-i - 1  )   
#            print(yt_.head()) 
            data = pd.concat ([yt4, yt_, yt1, yt2, yt3], axis =1)
            data. columns = ['yt4', 'yt_', 'yt1', 'yt2', 'yt3']
     
            data = data.dropna()
     
    
     
# target variable - closed price
            y = data ['yt_']
 
        
#       closed,   open,  high,   low    
            cols =['yt4', 'yt1', 'yt2', 'yt3']
            x = data [cols]
 
   
    
            scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
            x = np. array (x).reshape ((len( x) ,len(cols)))
            x = scaler_x.fit_transform (x)
 
    
            scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
            y = np.array (y).reshape ((len( y), 1))
            y = scaler_y.fit_transform (y)
 
 
 
     
            x_train = x [: train_end,:]
 
 
            x_test = x[ train_end: ,:]    
            y_train = y [: train_end] 
 
            args.y_train=y_train
 
            y_test = y[ train_end:]  
  
#        print(len(x_train), len(x_test), len(y_train), len(y_test))
                
            if (i == 0) :     
                prediction_data=[]
                for j in range (len(y_test) ) :
                    prediction_data.append (0)       
#        print(prediction_data, len(prediction_data)) 
 
 
            x_train = x_train.reshape (x_train. shape + (1,)) 
            x_test = x_test.reshape (x_test. shape + (1,))
 
            seed =2018
            np.random.seed (seed)
  
            fit0=lstm(cols)
            fit0.fit (x_train, y_train, batch_size =batch_size,epochs =10, shuffle = False)
            #  train_mse[i] = fit0.evaluate (x_train, y_train, batch_size =batch_size)
            #  test_mse[i] = fit0.evaluate (x_test, y_test, batch_size =batch_size)
            pred = fit0.predict (x_test) 
            pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
             # below is just fo i == 0
            if i==0:
                for j in range (len(pred)):
                    prediction_data[j] = pred[j] 
                   
#        print(prediction_data,len(prediction_data))
                
            forecast.iloc[i,k-1]=np.round(pred[-1],3)
#            print(forecast)
            upper.iloc[i,k-1]=np.round(pred[-1]*(1+bound_tuner),3)
            lower.iloc[i,k-1]=np.round(pred[-1]*(1-bound_tuner),3)
#        print(forecast)
          
            x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))           
###########      '''
    #  x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))

        prediction_data = np.asarray(prediction_data)
        prediction_data = prediction_data.ravel()
        for j in range (len(prediction_data)- 1):
            prediction_data[len(prediction_data) - j -1 ] =  prediction_data[len(prediction_data) - 1 - j - 1]
 
#        print(upper,lower)
    args.upper=upper
    args.lower=lower
    args.forecast= forecast    
def anomaly_check(args):
    df= args.query1
    h, w = df.shape[1]-1, args.steps_to_predict;
    alert=pd.DataFrame([[0 for x in range(w)] for y in range(h)]).T
    alert= pd.DataFrame(alert, columns = ["A_open", "A_high", "A_low", "A_close"])
    upper = args.upper
    lower = args.lower
    query2_1=args.query2
    query2= args.query2.iloc[:,1:]
    forecast=args.forecast
    print(alert,alert.shape)
    print("forecast",forecast,forecast.shape)
    print("upper", upper)
    print("lower", lower)
    print("closing", query2, query2.shape)

    for k in range( forecast.shape[1]):
        for i in range(forecast.shape[0]):
            try:
               # print(alert[i][k])
                if query2.iloc[i,k]<=lower.iloc[i,k]:
#                    print("no anomaly")
                    alert.iloc[i,k]='Alert, Going down'
                elif query2.iloc[i,k]>=upper.iloc[i,k]: 
#                    print("no anomaly")
                    alert.iloc[i,k]='Alert, Going up'
                else:
#                    print('alert')
                    alert.iloc[i,k]='Normal'
                print(lower.iloc[i,k],query2.iloc[i,k],upper.iloc[i,k])
                
            except:
                continue
    result = pd.concat([query2_1, alert,  upper, lower], axis=1, join_axes=[query2.index])
    result.to_gbq('pecten_dataset_dev.anomaly_detection_adidas',project_id='igenie-project',if_exists='append',private_key='/home/dileep/igenie-project-key.json')
#    print(result)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('max_date', help='Max_date')
    parser.add_argument("min_date",  help="min_date - format YYYY-MM-DD")
    args = parser.parse_args()
#    sys.path.insert(0, args.python_path)
    main(args)

