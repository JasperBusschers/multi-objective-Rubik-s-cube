import numpy as np
import csv
import pandas as pd
from pandas import ExcelWriter
import gym_Rubiks_Cube
import gym
from pandas import ExcelFile
import pickle



def write_list(triples, dir):
    with open(dir + ".csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')


def collect_data(dir, steps = 26, amount = 10000):
    env = gym.make("RubiksCube-v0")
    data = []
    states = []
    actions = []
    for i in range(0,amount):
        env.setScramble(1, 10, False)
        prev = env.reset()
        for j in range(0,steps):
            move = np.random.randint(0,11)
            obs = env.step(move)[0]
            if move < 6:
                move += 6
            else:
                move -= 6
            data.append([obs, move])
            states.append(obs)
            actions.append(move)
    with open('actions_2.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(actions, filehandle)
        with open('states_2.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(states, filehandle)
collect_data("stat-action-cube_1")