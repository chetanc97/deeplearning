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
score_requirment = 190
initial_games = 10000

def random_games():
    for episode in range(0 , 10):
        env.reset()
        prev = []
        for _ in range(0 , 10):
            env.render()
            action =  1
            print("Action\n")
            print(action)
            print(env.action_space)
            #> Discrete(2)
            print(env.observation_space)
            print(env.observation_space.high)
            print(env.observation_space.low)

            observation, reward, done , info  = env.step(action)
            print("1\n")
            if  (len(prev) > 0) and (observation[0] > prev[0]):
                print('loop')
                print(observation[0])
                print(prev[0])
                print('\n')

            prev = observation
            # print("2\n")
            # print(reward)
            # print("3\n")
            # print(done)
            # print("4\n")
            # print(info)
            if done:
                break

# random_games()
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation =[]
        second_prev_observation = []
        for _ in range(goal_steps):
            action =  random.randrange(0,3)
            observation , reward ,done,info = env.step(action)

            if (len(prev_observation) > 0) and (len(second_prev_observation) > 0):
                if ((observation[0] > prev_observation[0])  and  (prev_observation[0] < second_prev_observation[0] )):
                    game_memory.append([prev_observation , action])
                    score += 1

                if  ((observation[1] > prev_observation[1])  and  (prev_observation[1] < second_prev_observation[1] )):
                    score += 1
                if  (observation[1] > prev_observation[1]):
                    score += 1

            second_prev_observation = prev_observation
            prev_observation = observation

            #print('Aa1')
            #print(action)
            #print(score)
            if done:
                break
        if score >= score_requirment:
            accepted_scores.append(score)
            #print('her2')
            #print(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0 ,1 , 0]
                elif data[1] == 0:
                    output = [1 , 0 ,0]
                elif data[1] == 2:
                    output = [0 , 0 ,1]
                training_data.append([data[0] ,  output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy' , training_data_save)
    print('Avg accepted score ',  mean(accepted_scores))
    print('Median accepted score ' ,  median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network(input_size):
    network = input_data(shape = [None , input_size , 1] ,  name='input')

    network = fully_connected(network , 128 , activation ='relu')
    network = dropout(network , 0.8)

    network =  fully_connected(network , 256 , activation='relu')
    network = dropout(network , 0.8)

    network =  fully_connected(network , 512 , activation='relu')
    network = dropout(network , 0.8)
    #
    network =  fully_connected(network , 1024 , activation='relu')
    network = dropout(network , 0.8)

    network =  fully_connected(network , 512 , activation='relu')
    network = dropout(network , 0.8)

    network =  fully_connected(network , 256 , activation='relu')
    network = dropout(network , 0.8)

    network =  fully_connected(network , 128 , activation='relu')
    network = dropout(network , 0.8)

    network =  fully_connected(network , 3 , activation = 'softmax')
    network =  regression(network , optimizer = 'adam' , learning_rate = LR , loss= 'categorical_crossentropy' , name= 'targets')

    model = tflearn.DNN(network , tensorboard_dir = 'log')

    return model

def train_my_model(training_data ,model = False):
    X = np.array([i[0]  for i in  training_data ]).reshape(-1 , len(training_data[0][0] ) , 1  )
    y = [i[1] for i in training_data ]
    #print('targets\n')
    #print(y)
    if not model:
        model = neural_network(input_size = len(X[0]))

    model.fit({'input' : X } , {'targets' : y} ,  n_epoch = 3 , snapshot_step =500 ,  show_metric= True , run_id= 'openaistuff' )

    return model


training_data = initial_population()
model= train_my_model(training_data)


scores = []
choices = []

for game in range(15):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0 :
            action =random.randrange(0,3)
        else:
            action = np.argmax( model.predict(prev_obs.reshape(-1 ,  len(prev_obs) , 1)) [0] )
        choices.append(action)
        observation , reward, done , info  =  env.step(action)
        prev_obs = observation
        game_memory.append([prev_obs ,action])
        score+=reward

        if done:
            break
    scores.append(score)

print("Average score " , sum(scores) /  len(scores))
print("Choices 1 : {}  Choices 0 : {}  Choices 2 : {}"  .format(choices.count(1) /  len(choices) ,  choices.count(0) / len(choices) , choices.count(2)/len(choices) ) )
