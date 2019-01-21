from ast import literal_eval
import pickle
import numpy as np
import pandas as pd
class ExpertTraj:
    def __init__(self, env_name):
        #df = pd.read_excel('../GAIL_TEST/DATA/stat-action-cube_1.csv', sheetname=0)
        #data = pd.read_csv('../GAIL_TEST/DATA/stat-action-cube_1.csv', low_memory=False)
        self.exp_states = []
        self.exp_actions = []
        #print(data.shape)
        i =0
        with open('../GAIL_TEST/DATA/actions_2.data', 'rb') as f:
                self.exp_actions = pickle.load(f)
        with open('../GAIL_TEST/DATA/states_2.data', 'rb') as f:
                self.exp_states = pickle.load(f)
        print(self.exp_states[0])
        self.n_transitions = len(self.exp_actions)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action = [], []
        states = np.zeros([batch_size,54])
        actions = np.zeros([batch_size, 12])
        j=0
        for i in indexes:
            s = self.exp_states[i]
            a = self.exp_actions[i]
            states[j, :] = s
            actions[j, a] =1
        return states, actions