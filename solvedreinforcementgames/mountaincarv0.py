import gym
import random
import numpy as np
import tflearn
from  tflearn.layers.core  import  fully_connected ,  dropout ,  input_data
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env =  gym.make('MountainCar-v0')
env.reset()
goal_steps = 500
score_requirment = 50
initial_games = 10000

def random_games():
    for episode in range(0 , 30):
        env.reset()
        for _ in range(0 , 30):
            env.render()
            action =  1
            print("Action\n")
            print(action)
            print(env.action_space)
            #> Discrete(2)
            print(env.observation_space)
            observation, reward, done , info  = env.step(action)
            print("1\n")
            print(observation)
            print("2\n")
            print(reward)
            print("3\n")
            print(done)
            print("4\n")
            print(info)
            if done:
                break

#random_games()
def initial_games():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation =[]
        for _ in range(goal_steps):
            action.randrange(0,3)
            observation , reward ,done,info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation , action])
            prev_observation =observation
            score += reward
            if done:
                break
        if score >= score_requirment:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] = 1:
                    output = [0 ,1 , 0]
                elif data[1] = 0:
                    output = [1 , 0 ,1]
                elif data[1] = 2:
                    output = [0 , 0 ,1]
                training_data.append([data[0] ,  output])
        env.reset()
        scores.append(score)
