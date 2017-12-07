import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirment = 50
initial_games = 10000

def some_random_games_first():
    for episode in range(50):
        env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done :
                break

some_random_games_first()
