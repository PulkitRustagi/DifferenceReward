"""
Python implementation for Difference Reward structure for El Farol Bar Problem

"""
import math
import numpy as np
import matplotlib as plt
import random
import copy


class Agent:
    def __init__(self, num_nights, epsilon):
        self.choice = random.randrange(num_nights)
        self.DR = np.zeros(num_nights)
        self.epsilon = epsilon

    def make_choice(self):
        choices = list(range(len(self.DR)))
        # print("choices of nights: ", choices)
        idx = self.DR.argmax()
        choices.remove(idx)
        if random.uniform(0, 1) <= self.epsilon:
            idx = random.choice(choices)

        self.choice = idx


class Bar:
    def __init__(self, num_nights, capacity, num_of_people, epsilon, iterations):
        self.system_state = np.zeros(num_nights)
        self.num_of_nights = num_nights
        self.best_system_state = np.zeros(num_nights)
        self.agents = []
        self.G = 0.0
        self.iterations = iterations
        self.capacity = capacity
        self.num_of_agents = num_of_people
        for i in range(self.num_of_agents):
            self.agents.append(Agent(num_nights, epsilon))

    def make_choices(self):
        for i in range(self.num_of_agents):
            self.agents[i].make_choice()

    def update_system_state(self):
        self.system_state = np.zeros(self.num_of_nights)
        for i in self.agents:
            self.system_state[i.choice] += 1

    def update_global_reward(self):
        self.G = calc_global_reward(self.system_state, self.capacity)

    def update_local_reward(self):
        # sys_state = copy.copy(self.system_state)
        # print("Before Local update 1:", sys_state)
        for i in range(self.num_of_agents):
            sys_state = copy.copy(self.system_state)
            self.agents[i].DR[self.agents[i].choice] = difference_reward(sys_state, copy.copy(self.agents[i].choice),
                                                                         copy.copy(self.capacity))

        # print("After Local update 2:", sys_state)
        # print("After Local update 3:", self.system_state)


def difference_reward(system_state, agents_loc, capacity):
    G_z = calc_global_reward(system_state, capacity)
    sys = system_state[:]
    sys[agents_loc] = sys[agents_loc] - 1
    G_z_ = calc_global_reward(sys, capacity)
    # print("System State in DR: ", sys)
    DR = G_z - G_z_
    return DR


def calc_global_reward(system_state, capacity):
    G = 0
    for i in system_state:
        G += i * math.exp(-i / capacity)
    return G


'''
Do all the function calling sequentially

store config for max generated G

run iteration with diff epsilons (randomize differently?)
iterations = 1000
max_G = 0
for i in range(1000):
    if G>max_G:
        max_G = G
        store config
    choice
    global
    local
    
'''
# System parameters #
num_of_nights = 10  # k
Capacity = 10  # b
total_population = 100  # n
Epsilon = 0.1
iterations = 10000
G_max = 0.0
best_system_state = []
###################################
bar = Bar(num_of_nights, Capacity, total_population, Epsilon, iterations)

bar.make_choices()
bar.update_system_state()
bar.update_global_reward()
bar.update_local_reward()
G = copy.copy(bar.G)
system_state = copy.copy(bar.system_state)
print("\n----------------------------------------------")
print("Start System reward: ", G)
print("Start System state: ", system_state)
for i in range(1000):
    bar.make_choices()
    bar.update_system_state()
    bar.update_global_reward()
    bar.update_local_reward()
    if G_max <= bar.G:
        G_max = copy.copy(bar.G)
        best_system_state = copy.copy(bar.system_state)

    G = copy.copy(bar.G)
    system_state = copy.copy(bar.system_state)

# print("\n----------------------------------------------")
print("\nBest System reward: ", G_max)
print("Best System state: ", best_system_state)
print("\nEnd System reward: ", G)
print("End System state: ", system_state)
print("----------------------------------------------")
