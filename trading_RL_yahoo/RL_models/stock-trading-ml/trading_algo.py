import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from util import csv_to_dataset, history_points

def trade(symbol,days):
    model = load_model('technical_model.h5')
    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(symbol)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)
    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]
    
    
    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]
    
    unscaled_y_test = unscaled_y[n:]


    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    buys = []
    sells = []
    thresh = 0.1

    start = -days
    end = -1
    x = 1
    for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]])
        price_today = y_normaliser.inverse_transform(normalised_price_today)
        predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))
        delta = predicted_price_tomorrow - price_today
        if delta > thresh:
            buys.append((x, price_today[0][0]))
        elif delta < -thresh:
            sells.append((x, price_today[0][0]))
        x += 1
    # print(f"buys: {len(buys)}")
    # print(f"sells: {len(sells)}")


    def compute_earnings(buys_, sells_):
        purchase_amt = 10
        stock = 0
        balance = 0
        while len(buys_) > 0 and len(sells_) > 0:
            if buys_[0][0] < sells_[0][0]:
                # time to buy $10 worth of stock
                balance -= purchase_amt
                stock += purchase_amt / buys_[0][1]
                buys_.pop(0)
            else:
                # time to sell all of our stock
                balance += stock * sells_[0][1]
                stock = 0
                sells_.pop(0)
        # print(f"earnings: ${balance}")


    # we create new lists so we dont modify the original
    compute_earnings([b for b in buys], [s for s in sells])


    base = datetime.datetime.today()
    date_list = [(base - datetime.timedelta(days=200) + datetime.timedelta(days=x)).strftime("%d/%m/%Y") for x in range(0,days+1)]
    data = pd.DataFrame()

    data['Date'] = date_list[-len(unscaled_y_test):]
    data['Real_price'] = unscaled_y_test[-len(date_list):]
    data['Predicted_price'] = y_test_predicted[-len(date_list):]
    
    data.to_csv(symbol.split(".")[0] + '_output.csv', index=False)
    print(data.tail(1))


