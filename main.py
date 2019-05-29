import copy
import math
import random
import numpy as np
import time
import gym
import tqdm
from memory import ReplayMemory
from model import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt

from agent import *
env = gym.make('Pong-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# hyper params
ACTIONS = env.action_space.n
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
lr = 1e-4
INITIAL_MEMORY = 10
MEMORY_SIZE = 10 * INITIAL_MEMORY

agent = Agent(env)
rewards = agent.train(5000)




