import numpy as np
import pandas as pd
import os
import sys

file = sys.argv[1]
df = pd.read_csv(file)
print(df.head())
date = df.Date.values.tolist()

def abcd(trend, skip_loop = 25, ma = 7):
    ma = pd.Series(trend).rolling(ma).mean().values
    x = []
    for a in range(ma.shape[0]):
        for b in range(a, ma.shape[0], skip_loop):
            for c in range(b, ma.shape[0], skip_loop):
                for d in range(c, ma.shape[0], skip_loop):
                    if ma[b] > ma[a] and \
                    (ma[c] < ma[b] and ma[c] > ma[a]) \
                    and ma[d] > ma[b]:
                        x.append([a,b,c,d])
    x_np = np.array(x)
    ac = x_np[:,0].tolist() + x_np[:,2].tolist()
    bd = x_np[:,1].tolist() + x_np[:,3].tolist()
    ac_set = set(ac)
    bd_set = set(bd)
    signal = np.zeros(len(trend))
    buy = list(ac_set - bd_set)
    sell = list(list(bd_set - ac_set))
    signal[buy] = 1.0
    signal[sell] = -1.0
    return signal

signal = abcd(df['Close'])

def buy_stock(
    real_movement,date,
    signal,
    initial_money = 10000,
    max_buy = 1,
    max_sell = 1,
):
    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 10000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """
    starting_money = initial_money
    states_sell = []
    states_buy = []
    states_money = []
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
            states_buy.append(0)
        return initial_money, current_inventory
    
    for i in range(real_movement.shape[0]):
        state = signal[i]
        if state == 1:
            initial_money, current_inventory = buy(
                i, initial_money, current_inventory
            )
            states_buy.append(i)
            df1 = pd.DataFrame({'Date': date[i], 'Close': [real_movement[i]], 'RESULT': ['Buy']})
            if not os.path.isfile(file_path):
                df1.to_csv(file_path, index=False)
            else:
                df1.to_csv(file_path, index=False, mode='a', header=False)
                    
        elif state == -1:
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
            states_sell.append(i)
            df2 = pd.DataFrame({'Date': date[i], 'Close': [real_movement[i]], 'RESULT': ['Sell']})
            if not os.path.isfile(file_path):
                df2.to_csv(file_path, index=False)
            else:
                    df2.to_csv(file_path, index=False, mode='a', header=False)
                    
        else:
            df3 = pd.DataFrame({'Date': date[i], 'Close': [real_movement[i]], 'RESULT': ['Hold']})
            if not os.path.isfile(file_path):
                df3.to_csv(file_path, index=False)
            else:
                df3.to_csv(file_path, index=False, mode='a', header=False)       
            print(
                    'day %d, hold UNIT at price %f,  total balance %f,'
                    % (i, real_movement[i], initial_money)
                )
            states_money.append(initial_money)
        
            
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    print(
                '\ntotal gained %f, total investment %f %%'
                % (initial_money - starting_money, invest)
            )

file_path = "abcd_{}.csv".format(file)

buy_stock(df.Close,date,signal)

result = pd.read_csv(file_path)
print(result.tail(5))
