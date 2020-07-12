import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from bayes_opt import BayesianOptimization
import os, sys

# from crypto_data_loader import get_candles

sns.set()

use_csv = True


def get_state(data, t, n):
    d = t - n + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[: t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])


class Deep_Evolution_Strategy:
    inputs = None

    def __init__(
            self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=5, print_every=1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                        w
                        + self.learning_rate
                        / (self.population_size * self.sigma)
                        * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    def __init__(
            self,
            population_size,
            sigma,
            learning_rate,
            model,
            money,
            max_buy,
            max_sell,
            skip,
            window_size,
    ):
        self.window_size = window_size
        self.skip = skip
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.model = model
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), int(buy[0])

    def get_reward(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights
        state = get_state(close0, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, l0):
            action, buy = self.act(state)
            next_state = get_state(close0, t + 1, self.window_size + 1)
            if action == 1 and initial_money >= close0[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * close0[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * close0[t]
                initial_money += total_sell

            state = next_state
        return ((initial_money - starting_money) / starting_money) * 100

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        initial_money = self.initial_money
        state = get_state(close1, 0, self.window_size + 1)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        profit = 0
        loss = 0
        for t in range(0, l1):
            action, buy = self.act(state)
            next_state = get_state(close1, t , self.window_size + 1)
            if action == 1 and initial_money >= close1[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * close1[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
                print(
                    'day %d: buy %d units at price %f, total balance %f'
                    % (t+1, buy_units, total_buy, initial_money)
                )
                df1 = pd.DataFrame({'Date': date1[t + 1], 'Close': [close1[t + 1]], 'RESULT': ['Buy']})
                if not os.path.isfile('evolution-bayesian.csv'):
                    df1.to_csv('evolution-bayesian.csv', index=False)
                else:
                    df1.to_csv('evolution-bayesian.csv', index=False, mode='a', header=False)
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                if sell_units < 1:
                    continue
                quantity -= sell_units
                total_sell = sell_units * close1[t]
                initial_money += total_sell
                states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100

                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                    % (t+1, sell_units, total_sell, invest, initial_money)
                )
                df2 = pd.DataFrame({'Date': date1[t + 1], 'Close': [close1[t + 1]], 'RESULT': ['Sell']})
                if not os.path.isfile('evolution-bayesian.csv'):
                    df2.to_csv('evolution-bayesian.csv', index=False)
                else:
                    df2.to_csv('evolution-bayesian.csv', index=False, mode='a', header=False)
                if invest > 0:
                    profit += 1
                else:
                    loss += 1


            else:
                print(
                    'day %d, hold UNIT at price %f,  total balance %f,'
                    % (t + 1, close1[t + 1], initial_money)
                )
                df3 = pd.DataFrame({'Date': date1[t + 1], 'Close': [close1[t + 1]], 'RESULT': ['Hold']})
                if not os.path.isfile('evolution-bayesian.csv'):
                    df3.to_csv('evolution-bayesian.csv', index=False)
                else:
                    df3.to_csv('evolution-bayesian.csv', index=False, mode='a', header=False)


            state = next_state
        fi = pd.read_csv('evolution-bayesian.csv')
        print(fi.tail(2))

        invest = ((initial_money - starting_money) / starting_money) * 100
        ratio = (profit / (loss + profit)) * 100
        print(
            '\ntotal gained %f, total investment %f %%'
            % (initial_money - starting_money, invest)
        )
        print(
            'total wins %d , total losses %d, accuracy ratio: %f'
            % (profit, loss, ratio)
        )
        print('left in inventory: %d' % (len(inventory)))
        print(inventory)




def best_agent(
        window_size, skip, population_size, sigma, learning_rate, size_network
):
    model = Model(window_size, size_network, 3)
    agent = Agent(
        population_size,
        sigma,
        learning_rate,
        model,
        10000,
        5,
        5,
        skip,
        window_size,
    )
    try:
        agent.fit(1, 1)
        return agent.es.reward_function(agent.es.weights)
    except:
        return 0


def find_best_agent(
        window_size, skip, population_size, sigma, learning_rate, size_network
):
    global accbest
    param = {
        'window_size': int(np.around(window_size)),
        'skip': int(np.around(skip)),
        'population_size': int(np.around(population_size)),
        'sigma': max(min(sigma, 1), 0.0001),
        'learning_rate': max(min(learning_rate, 0.5), 0.000001),
        'size_network': int(np.around(size_network)),
    }
    print('\nSearch parameters %s' % (param))
    investment = best_agent(**param)
    print('stop after 100 iteration with investment %f' % (investment))
    if investment > accbest:
        costbest = investment
    return investment


##--------------------------------------------------
# ##Settings (Later as argv)
# exchange = 'bitmex'
# symbol = 'BTC/USD'
# start_date = '2018-01-01T00:00:00Z'
# timeframe = '1d'

# ---------------------------------------------------
##Get Data
if use_csv:
    file = sys.argv[1]
    df = pd.read_csv(file)
# else:
#     df = get_candles(exchange, 3, symbol, timeframe, start_date, 25)  # df = pd.read_csv('../dataset/GOOG-year.csv')
print(df)

df = pd.read_csv(file)
df.head()
df0 = df.iloc[:503,:]
# print('dfo0')
# print(df0.head())
df1 = df.iloc[503:,:]
# print('dfo1')
# print(df1.head())
close0 = df0.Close.values.tolist()
close1 = df1.Close.values.tolist()
date1 = df1.Date.values.tolist()
l0 = len(close0) - 1
l1 = len(close1) - 1


close = df.Close.values.tolist()


window_size = 30
model = Model(window_size, 100, 3)

initial_money = 10000
starting_money = initial_money
len_close = len(close) - 1
l=len(close) - 1
weight = model
skip = 1

state = get_state(close0, 0, window_size + 1)
inventory = []
quantity = 0

max_buy = 5
max_sell = 5

##----------------------------------------------------
## Bayesian Stuff
accbest = 0.0
NN_BAYESIAN = BayesianOptimization(
    find_best_agent,
    {
        'window_size': (2, 50),  # standard: 2,50
        'skip': (1, 15),  # standard: 1,15
        'population_size': (1, 50),  # standard: 1,50
        'sigma': (0.01, 0.99),
        'learning_rate': (0.000001, 0.49),  # standard: 0.000001 , 0.49
        'size_network': (10, 100),  # standard: 10,1000
    },
)
NN_BAYESIAN.maximize(init_points=5, n_iter=5, acq='ei', xi=0.0)  # standard: init_points=30 n_iter=50

print('----------------------------------------------')
print(NN_BAYESIAN.max)

params = NN_BAYESIAN.max['params']
# ______________________________________________________________________________
# ------------------------------------------------------------------------------
best_agent(int(params['window_size']), int(params['skip']),
           int(params['population_size']), params['sigma'],
           params['learning_rate'], int(params['size_network']))

model = Model(int(params['window_size']), int(params['size_network']), 3)
agent = Agent(int(params['population_size']), params['sigma'], params['learning_rate'], model, 10000, 5, 5,
              int(params['skip']), int(params['window_size']))

agent.fit(iterations=1, checkpoint=1)

agent.buy()