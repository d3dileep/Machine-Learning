import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import os
import time
import sys
sns.set()
tf.disable_v2_behavior()

df = pd.read_csv('AAPL.csv')
df.head()

from collections import deque
import random


initial_money = 10000
window_size = 30
skip = 1

novelty_search_threshold = 6
novelty_log_maxlen = 1000
backlog_maxsize = 500
novelty_log_add_amount = 3

class neuralnetwork:
	def __init__(self, id_, hidden_size = 128):
		self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
		self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
		self.fitness = 0
		self.last_features = None
		self.id = id_

def relu(X):
	return np.maximum(X, 0)
    
def softmax(X):
	e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
	return e_x / np.sum(e_x, axis=-1, keepdims=True)

def feed_forward(X, nets):
	a1 = np.dot(X, nets.W1)
	z1 = relu(a1)
	a2 = np.dot(z1, nets.W2)
	return softmax(a2)

class NeuroEvolution:
	def __init__(self, population_size, mutation_rate, model_generator,state_size, window_size, trend, skip, initial_money):
		self.population_size = population_size
		self.mutation_rate = mutation_rate
		self.model_generator = model_generator
		self.state_size = state_size
		self.window_size = window_size
		self.half_window = window_size // 2
		self.trend = trend
		self.skip = skip
		self.initial_money = initial_money
		self.generation_backlog = []
		self.novel_backlog = []
		self.novel_pop = []
        
	def _initialize_population(self):
		self.population = []
		for i in range(self.population_size):
			self.population.append(self.model_generator(i))
    
	def _memorize(self, q, i, limit):
		q.append(i)
		if len(q) > limit:
			q.pop()
	def mutate(self, individual, scale=1.0):
		mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
		individual.W1 += np.random.normal(loc=0, scale=scale, size=individual.W1.shape) * mutation_mask
		mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
		individual.W2 += np.random.normal(loc=0, scale=scale, size=individual.W2.shape) * mutation_mask
		return individual
    
	def inherit_weights(self, parent, child):
		child.W1 = parent.W1.copy()
		child.W2 = parent.W2.copy()
		return child
    
	def crossover(self, parent1, parent2):
		child1 = self.model_generator((parent1.id+1)*10)
		child1 = self.inherit_weights(parent1, child1)
		child2 = self.model_generator((parent2.id+1)*10)
		child2 = self.inherit_weights(parent2, child2)
		# first W
		n_neurons = child1.W1.shape[1]
		cutoff = np.random.randint(0, n_neurons)
		child1.W1[:, cutoff:] = parent2.W1[:, cutoff:].copy()
		child2.W1[:, cutoff:] = parent1.W1[:, cutoff:].copy()
		# second W
		n_neurons = child1.W2.shape[1]
		cutoff = np.random.randint(0, n_neurons)
		child1.W2[:, cutoff:] = parent2.W2[:, cutoff:].copy()
		child2.W2[:, cutoff:] = parent1.W2[:, cutoff:].copy()
		return child1, child2
    
	def get_state(self, t):
		window_size = self.window_size + 1
		d = t - window_size + 1
		block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
		res = []
		for i in range(window_size - 1):
			res.append(block[i + 1] - block[i])
		return np.array([res])
    
	def act(self, p, state):
		logits = feed_forward(state, p)
		return np.argmax(logits, 1)[0]
    
	def buy(self, individual):
		initial_money = self.initial_money
		starting_money = initial_money
		state = self.get_state(0)
		inventory = []
		states_sell = []
		states_buy = []
		for t in range(0, len(self.trend) - 1, self.skip):
			action = self.act(individual, state)
			next_state = self.get_state(t + 1)
			if action == 1 and starting_money >= self.trend[t]:
				inventory.append(self.trend[t])
				initial_money -= self.trend[t]
				states_buy.append(t)
				print('day %d: buy 1 unit at price %f, total balance %f'% (t, self.trend[t], initial_money))
				df1 = pd.DataFrame({'Date': date1[t+1],'Close': [close[t+1]], 'RESULT': ['Buy']})
				if not os.path.isfile('22.csv'):
					df1.to_csv('22.csv', index=False)
				else:
					df1.to_csv('22.csv', index=False, mode='a', header=False)
			elif action == 2 and len(inventory):
				bought_price = inventory.pop(0)
				initial_money += self.trend[t]
				states_sell.append(t)
				try:
					invest = ((self.trend[t] - bought_price) / bought_price) * 100
				except:
					invest = 0
				print('day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'% (t, self.trend[t], invest, initial_money))
				df2 = pd.DataFrame({'Date': date1[t+1], 'Close': [close[t+1]], 'RESULT': ['Sell']})
				if not os.path.isfile('22.csv'):
					df2.to_csv('22.csv', index=False)
				else:
					df2.to_csv('22.csv', index=False, mode='a', header=False)
			else:
				print('day %d, hold UNIT at price %f,  total balance %f,'% (t+1, close[t+1], initial_money))
				df3 = pd.DataFrame({'Date': date1[t+1], 'Close': [close[t+1]], 'RESULT': ['Hold']})
				if not os.path.isfile('22.csv'):
					df3.to_csv('22.csv', index=False)
				else:
					df3.to_csv('22.csv', index=False, mode='a', header=False)
			state = next_state
		fi = pd.read_csv('22.csv')
		print(fi.tail(2))
		invest = ((initial_money - starting_money) / starting_money) * 100
		total_gains = initial_money - starting_money
		return states_buy, states_sell, total_gains, invest
    
	def calculate_fitness(self):
		for i in range(self.population_size):
			initial_money = self.initial_money
			starting_money = initial_money
			state = self.get_state(0)
			inventory = []
			for t in range(0, len(self.trend) - 1, self.skip):
				action = self.act(self.population[i], state)
				next_state = self.get_state(t + 1)
				if action == 1 and starting_money >= self.trend[t]:
					inventory.append(self.trend[t])
					starting_money -= self.trend[t]
				elif action == 2 and len(inventory):
					bought_price = inventory.pop(0)
					starting_money += self.trend[t]

				state = next_state
			invest = ((starting_money - initial_money) / initial_money) * 100	
			self.population[i].fitness = invest
			self.population[i].last_features = self.population[i].W2.flatten()
    
	def evaluate(self, individual, backlog, pop, k = 4):
		score = 0
		if len(backlog):
			x = np.array(backlog)
			nn = NearestNeighbors(n_neighbors = k, metric = 'euclidean').fit(np.array(backlog))
			d, _ = nn.kneighbors([individual])
			score += np.mean(d)
		if len(pop):
			nn = NearestNeighbors(n_neighbors = k, metric = 'euclidean').fit(np.array(pop))
			d, _ = nn.kneighbors([individual])
			score += np.mean(d)
		return score
    
	def evolve(self, generations=20, checkpoint= 5):
		self._initialize_population()
		n_winners = int(self.population_size * 0.4)
		n_parents = self.population_size - n_winners
		for epoch in range(generations):
			self.calculate_fitness()
			scores = [self.evaluate(p.last_features, self.novel_backlog, self.novel_pop) for p in self.population]
			sort_fitness = np.argsort(scores)[::-1]
			self.population = [self.population[i] for i in sort_fitness]
			fittest_individual = self.population[0]
			if (epoch+1) % checkpoint == 0:
				print('epoch %d, fittest individual %d with accuracy %f'%(epoch+1, sort_fitness[0],fittest_individual.fitness))
			next_population = [self.population[i] for i in range(n_winners)]
			total_fitness = np.sum([np.abs(i.fitness) for i in self.population])
			parent_probabilities = [np.abs(i.fitness / total_fitness) for i in self.population]
			parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)
			for p in next_population:
				if p.last_features is not None:
					self._memorize(self.novel_pop, p.last_features, backlog_maxsize)
					if np.random.randint(0,10) < novelty_search_threshold:
						self._memorize(self.novel_backlog, p.last_features, novelty_log_maxlen)
                        
			for i in np.arange(0, len(parents), 2):
				child1, child2 = self.crossover(parents[i], parents[i+1])
				next_population += [self.mutate(child1), self.mutate(child2)]
			self.population = next_population
            
			if np.random.randint(0,10) < novelty_search_threshold:
				pop_sorted = sorted(self.population, key=lambda p: p.fitness, reverse=True)
				self.generation_backlog.append(pop_sorted[0])
				print('novel add fittest, score: %f, backlog size: %d'%(pop_sorted[0].fitness,len(self.generation_backlog)))
				generation_backlog_temp = self.generation_backlog
				if len(self.generation_backlog) > backlog_maxsize:
					generation_backlog_temp = random.sample(generation_backlog, backlog_maxsize)
				for p in generation_backlog_temp:
					if p.last_features is not None:
						self._memorize(self.novel_backlog, p.last_features, novelty_log_maxlen)
                        
		return fittest_individual

date1 = df.Date.values.tolist()

close = df.Close.values.tolist()

population_size = 100
generations = 100
mutation_rate = 0.1
neural_evolve = NeuroEvolution(population_size, mutation_rate, neuralnetwork,window_size, window_size, close, skip, initial_money)

fittest_nets = neural_evolve.evolve(30)

states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)


