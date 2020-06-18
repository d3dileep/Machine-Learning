import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time

sns.set()

solution = np.random.randn(100)
w = np.random.randn(100)


def f(w):
    return -np.sum(np.square(solution - w))


def get_state(data, t, n):
    d = t - n + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[: t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])


class Deep_Evolution_Strategy:
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


def act(model, sequence):
    decision, buy = model.predict(np.array(sequence))
    return np.argmax(decision[0]), int(buy[0])


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(
            self, model: object, money: object, max_buy: object, max_sell: object, close: object, window_size: object,
            skip: object
    ) -> object:
        self.window_size = window_size
        self.skip = skip
        self.close = close
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
        len_close = len(self.close) - 1

        self.model.weights = weights
        state = get_state(close0, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, l0, self.skip):
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
        len_close = len(close1) - 1
        state = get_state(close1, 0, self.window_size + 1)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        for t in range(0, l1, self.skip):
            action, buy = self.act(state)
            next_state = get_state(close1, t + 1, self.window_size + 1)
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
                    % (t, buy_units, total_buy, initial_money)
                )
                df1 = pd.DataFrame({'Date': date1[t], 'Close': [close1[t]], 'RESULT': ['Buy']})
                if not os.path.isfile('updated-NES-google.csv'):
                    df1.to_csv('updated-NES-google.csv', index=False)
                else:
                    df1.to_csv('updated-NES-google.csv', index=False, mode='a', header=False)
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
                    % (t, sell_units, total_sell, invest, initial_money)
                )
                df2 = pd.DataFrame({'Date': date1[t], 'Close': [close1[t]], 'RESULT': ['Sell']})
                if not os.path.isfile('updated-NES-google.csv'):
                    df2.to_csv('updated-NES-google.csv', index=False)
                else:
                    df2.to_csv('updated-NES-google.csv', index=False, mode='a', header=False)

            state = next_state

        invest = ((initial_money - starting_money) / starting_money) * 100
        print(
            '\ntotal gained %f, total investment %f %%'
            % (initial_money - starting_money, invest)
        )


npop = 50
sigma = 0.1
alpha = 0.001

for i in range(5000):

    if (i + 1) % 1000 == 0:
        print(
            'iter %d. w: %s, solution: %s, reward: %f'
            % (i + 1, str(w[-1]), str(solution[-1]), f(w))
        )
    N = np.random.randn(npop, 100)
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma * N[j]
        R[j] = f(w_try)

    A = (R - np.mean(R)) / np.std(R)
    w = w + alpha / (npop * sigma) * np.dot(N.T, A)

sigma = 0.1
N = np.random.randn(npop, 100)
individuals = []
for j in range(2):
    individuals.append(w + sigma * N[j])

file = sys.argv[1]

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
get_state(close, 0, 10)

get_state(close, 1, 10)

get_state(close, 2, 10)


window_size = 30
model = Model(window_size, 500, 3)

initial_money = 10000
starting_money = initial_money
len_close = len(close) - 1
weight = model
skip = 1

state = get_state(close0, 0, window_size + 1)
inventory = []
quantity = 0

max_buy = 5
max_sell = 5

for t in range(0, l0, skip):
    action, buy = act(weight, state)
    next_state = get_state(close0, t + 1, window_size + 1)
    if action == 1 and initial_money >= close0[t]:
        if buy < 0:
            buy = 1
        if buy > max_buy:
            buy_units = max_buy
        else:
            buy_units = buy
        total_buy = buy_units * close0[t]
        initial_money -= total_buy
        inventory.append(total_buy)
        quantity += buy_units
    elif action == 2 and len(inventory) > 0:
        if quantity > max_sell:
            sell_units = max_sell
        else:
            sell_units = quantity
        quantity -= sell_units
        total_sell = sell_units * close0[t]
        initial_money += total_sell

    state = next_state
((initial_money - starting_money) / starting_money) * 100

model = Model(input_size=window_size, layer_size=500, output_size=3)
agent = Agent(
    model=model,
    money=10000,
    max_buy=5,
    max_sell=5,
    close=close,
    window_size=window_size,
    skip=1,
)

agent.fit(iterations=500, checkpoint=10)

agent.buy()