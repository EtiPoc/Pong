import random
from utils import *
import numpy as np


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.initial_memory = self.capacity/4

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def fill_memory(self, env):
        obs = rgb2gray(env.reset()).reshape(1, 80, 80)
        num_pos = 0
        game = []
        for i in range(3):
            obs = np.append(obs, rgb2gray(env.step(0)[0]), 0)
        while self.__len__() < self.initial_memory*0.9:
            obs = rgb2gray(env.reset())
            for i in range(3):
                obs = np.append(obs, rgb2gray(env.step(0)[0]), 0)
            game = []
            while not env.ale.game_over():
                action = random.randint(0, 5)
                new_obs = env.step(action)
                reward, terminal = new_obs[1], new_obs[2]
                new_obs = np.append(obs[1:], rgb2gray(new_obs[0]), 0)
                game.append(Transition(obs, action, new_obs, reward, terminal))

                # compute the discounted reward for the whole episode
                if reward !=0:
                    pos = len(game)-2
                    while game[pos][3] == 0 and pos >= 0:
                        element = game[pos]
                        new_element = Transition(element[0], element[1], element[2], reward*0.99**(len(game)-1-pos), element[4])
                        game[pos] = new_element
                        pos -= 1
                obs = new_obs
            for i in range(len(game)):
                transition = game.pop(0)
                self.push(transition[0], transition[1], transition[2], transition[3], transition[4])

        count = 0
        for i in range(self.__len__()):
            if self.memory[i][3] > 0:
                count += 1
        print(count)
        # same but add only examples with positive rewards
        while count < self.initial_memory/10:
            obs = rgb2gray(env.reset())
            for i in range(3):
                obs = np.append(obs, rgb2gray(env.step(0)[0]), 0)
            game = []
            while not env.ale.game_over():
                action = random.randint(0, 5)
                new_obs = env.step(action)
                reward, terminal = new_obs[1], new_obs[2]
                new_obs = np.append(obs[1:], rgb2gray(new_obs[0]), 0)

                game.append(Transition(obs, action, new_obs, reward, terminal))

                # Artificially increase the number of positive rewards by assigning r=1 to the 10 previous moves of actual r=1
                if reward == 1:
                    num_pos += 1
                    pos = len(game)-2
                    while game[pos][3] == 0 and pos >= 0:
                        element = game[pos]
                        new_element = Transition(element[0], element[1], element[2], 1.0*0.99**(len(game)-1-pos), element[4])
                        game[pos] = new_element
                        pos -= 1
                    game = game[pos+1:]
                    for i in range(len(game)):
                        transition = game.pop(0)
                        self.push(transition[0], transition[1], transition[2], transition[3], transition[4])
                    game = []
                obs = new_obs

                count = 0
                for i in range(self.__len__()):
                    if self.memory[i][3] > 0:
                        count += 1



    def prioritized_sample(self, batch_size):
        pass

    def __len__(self):
        return len(self.memory)
