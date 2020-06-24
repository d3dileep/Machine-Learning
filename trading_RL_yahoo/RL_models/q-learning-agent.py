import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import os
import sys
import matplotlib.pyplot as plt
from collections import deque
import random
import seaborn as sns
sns.set()
tf.compat.v1.disable_eager_execution()


def get_state(data, t, n):
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])


class Agent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen = 1000)
        self.inventory = []

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.Y = tf.placeholder(tf.float32, [None, self.action_size])
        feed = tf.layers.dense(self.X, 64, activation = tf.nn.relu)
        feed = tf.layers.dense(feed, 32, activation = tf.nn.relu)
        feed = tf.layers.dense(feed, 8, activation = tf.nn.relu)
        self.logits = tf.layers.dense(feed, self.action_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(
            self.cost
        )
        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(
            self.sess.run(self.logits, feed_dict = {self.X: state})[0]
        )

    def replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        replay_size = len(mini_batch)
        X = np.empty((replay_size, self.state_size))
        Y = np.empty((replay_size, self.action_size))
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        Q = self.sess.run(self.logits, feed_dict = {self.X: states})
        Q_new = self.sess.run(self.logits, feed_dict = {self.X: new_states})
        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i]
            target[action] = reward
            if not done:
                target[action] += self.gamma * np.amax(Q_new[i])
            X[i] = state
            Y[i] = target
        cost, _ = self.sess.run(
            [self.cost, self.optimizer], feed_dict = {self.X: X, self.Y: Y}
        )
        # print('cost: %f'%(cost))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

file = sys.argv[1]
df = pd.read_csv(file)
print(df.head())
close = df.Close.values.tolist()
df0 = df.iloc[:503,:]
df1 = df.iloc[503:,:]
# print('dfo0')
# print(df0.head())
# print('dfo1')
# print(df1.head())
close0 = df0.Close.values.tolist()
close1 = df1.Close.values.tolist()
date1 = df1.Date.values.tolist()
window_size = 10
skip = 4
l = len(close) - 1
batch_size = 32
agent = Agent(window_size)
epoch = 5

for e in range(epoch):
    state = get_state(close0, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(0, len(close0) - 1, skip):
        action = agent.act(state)
        next_state = get_state(close0, t + 1, window_size + 1)
        done = True
        reward = -10

        if action == 1:
            agent.inventory.append(close0[t])
            # print("Buy on %d for %f"%(t,close[t]))

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(close0[t] - bought_price, 0)
            done = False
            total_profit += close0[t] - bought_price
            # print("Sell on %d for %f, Profit %f"%(t,close[t],close[t] - bought_price))

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print('epoch %d, total profit %f' % (e + 1, total_profit))


state = get_state(close1, 0, window_size + 1)
initial_money = 10000
starting_money = initial_money
states_sell = []
states_buy = []
agent.inventory = []

for t in range(0, len(close1) - 1, skip):
    action = agent.act(state)
    next_state = get_state(close1, t + 1, window_size + 1)
    if action == 1 and initial_money >= close1[t]:
        agent.inventory.append(close1[t])
        initial_money -= close1[t]
        states_buy.append(t)

        print(
            'day %d: buy UNIT at price %f, total balance %f'
            % (t, close1[t], initial_money)
        )
        df1 = pd.DataFrame({'Date': date1[t], 'Close': [close1[t]],'RESULT': ['Buy'] })
        if not os.path.isfile('q-learning-agent.csv'):
            df1.to_csv('q-learning-agent.csv', index=False)
        else:
            df1.to_csv('q-learning-agent.csv', index=False, mode='a', header=False)
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        initial_money += close1[t]
        states_sell.append(t)
        try:
            invest = ((close1[t] - bought_price) / bought_price) * 100
        except:
            invest = 0
        print(
            'day %d, sell UNIT at price %f, investment %f %%, total balance %f,'
            % (t, close1[t], invest, initial_money)
        )
        df2 = pd.DataFrame({'Date': date1[t], 'Close': [close1[t]],'RESULT': ['Sell'] })
        if not os.path.isfile('q-learning-agent.csv'):
            df2.to_csv('q-learning-agent.csv', index=False)
        else:
            df2.to_csv('q-learning-agent.csv', index=False, mode='a', header=False)
    state = next_state

invest = ((initial_money - starting_money) / starting_money) * 100
print(
    '\ntotal gained %f, total investment %f %%'
    % (initial_money - starting_money, invest)
)
