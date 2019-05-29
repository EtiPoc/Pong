from model import DQN
from memory import ReplayMemory
import torch
import random
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import *
from tqdm import tqdm

MEMORY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99


class Agent:
    def __init__(self, env, exploration_rate=1, exploration_decay=0.9999, explore=True):
        self.action_space = env.action_space.n
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.memory.fill_memory(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(4, self.action_space).float().to(self.device)
        self.env = env
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.explore = explore
        self.model_optim = optim.Adam(self.dqn.parameters(), lr=1e-4)

    def get_action(self, obs):
        if self.exploration_rate>random.random() and self.explore:
            action = random.randint(0, self.action_space-1)
        else:
            obs = torch.tensor(obs, device=self.device).reshape(1, 4, 80, 80).float()
            action = self.dqn(obs).argmax().tolist()
        return action

    def train(self, num_episodes):
        num_steps = 0
        running_loss = 0
        loss = nn.MSELoss()

        episode_rewards = []
        for episode in tqdm(range(num_episodes)):
            obs = rgb2gray(self.env.reset()).reshape(1, 80, 80)
            for i in range(3):
                obs = np.append(obs, rgb2gray(self.env.step(0)[0]), 0)

            terminal = False
            episode_reward = 0
            while not terminal:
                action = self.get_action(obs)
                result = self.env.step(action)

                terminal = result[2]
                new_obs = np.append(obs[1:], rgb2gray(result[0]), 0)
                reward = result[1]
                if reward > 0:
                    print(episode, reward)
                episode_reward += reward

                self.memory.push(obs, action, new_obs, reward, terminal)
                batch = self.memory.sample(BATCH_SIZE)
                observations, y = self.process_batch(batch)
                num_steps += 1

                outputs = self.dqn(observations)
                episode_loss = loss(outputs, y)
                self.model_optim.zero_grad()
                episode_loss.backward()
                self.model_optim.step()
                running_loss += episode_loss.item()

                if num_steps % 1000 == 0:  # print every 2000 mini-batches
                    print(num_steps)

            episode_rewards.append(episode_reward)
            if self.exploration_rate > 0.1:
                self.exploration_rate *= self.exploration_decay
        return episode_rewards

    def process_batch(self, batch):
        observations = [batch[i][0] for i in range(len(batch))]
        observations = torch.tensor(np.array(observations)).reshape((BATCH_SIZE, 4, 80, 80)).float().to(self.device)

        next_observations = [batch[i][2] for i in range(len(batch))]
        next_observations = torch.tensor(np.array(next_observations)).reshape((BATCH_SIZE, 4, 80, 80)).float().to(self.device)

        maxs = self.dqn(next_observations)
        maxs = maxs.max(1).values.float().to(self.device)

        rewards = [batch[i][3] for i in range(len(batch))]
        rewards = torch.tensor(rewards).float().to(self.device)

        terminals = [~batch[i][4] for i in range(len(batch))]
        terminals = torch.tensor(terminals).float().to(self.device)

        maxs = -maxs * terminals

        y = self.dqn(observations)
        Qs = rewards + GAMMA * maxs

        for i in range(len(batch)):
            y[i, batch[i][1]] = Qs[i]

        return observations, y


