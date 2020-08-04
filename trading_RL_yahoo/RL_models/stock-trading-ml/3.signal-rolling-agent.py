import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
sns.set()

df = pd.read_csv('AAPL.csv')
df.head()


def buy_stock(
    real_movement,
    delay=5,
    initial_state=1,
    initial_money=10000,
    max_buy=1,
    max_sell=1,
):
    starting_money = initial_money
    delay_change_decision = delay
    current_decision = 0
    state = initial_state
    current_val = real_movement[0]
    states_sell = []
    states_buy = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement[i])
            )
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * real_movement[i], initial_money)
            )
            df1 = pd.DataFrame(
                {'Date': date1[i+1], 'Close': [close[i+1]], 'RESULT': ['Buy']})
            if not os.path.isfile('3.csv'):
                df1.to_csv('3.csv', index=False)
            else:
                df1.to_csv('3.csv', index=False, mode='a', header=False)
            states_buy.append(0)
        return initial_money, current_inventory

    if state == 1:
        initial_money, current_inventory = buy(
            0, initial_money, current_inventory
        )

    for i in range(1, real_movement.shape[0], 1):
        if real_movement[i] < current_val and state == 0:
            if current_decision < delay_change_decision:
                current_decision += 1
            else:
                state = 1
                initial_money, current_inventory = buy(
                    i, initial_money, current_inventory
                )
                current_decision = 0
                states_buy.append(i)
        if real_movement[i] > current_val and state == 1:
            if current_decision < delay_change_decision:
                current_decision += 1
            else:
                state = 0

                if current_inventory == 0:
                    print('day %d: cannot sell anything, inventory 0' % (i))
                else:
                    if current_inventory > max_sell:
                        sell_units = max_sell
                    else:
                        sell_units = current_inventory
                    current_inventory -= sell_units
                    total_sell = sell_units * real_movement[i]
                    initial_money += total_sell
                    try:
                        invest = (
                            (real_movement[i] - real_movement[states_buy[-1]])
                            / real_movement[states_buy[-1]]
                        ) * 100
                    except:
                        invest = 0
                    print(
                        'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                        % (i, sell_units, total_sell, invest, initial_money)
                    )
                    df2 = pd.DataFrame(
                        {'Date': date1[i+1], 'Close': [close[i+1]], 'RESULT': ['Sell']})
                    if not os.path.isfile('3.csv'):
                        df2.to_csv('3.csv', index=False)
                    else:
                        df2.to_csv('3.csv', index=False,
                                   mode='a', header=False)

                current_decision = 0
                states_sell.append(i)
        current_val = real_movement[i]
    fi = pd.read_csv('3.csv')
    print(fi.tail(2))
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    return states_buy, states_sell, total_gains, invest


date1 = df.Date.values.tolist()

close = df.Close.values.tolist()
states_buy, states_sell, total_gains, invest = buy_stock(df.Close, initial_state=1,
                                                         delay=4, initial_money=10000)
