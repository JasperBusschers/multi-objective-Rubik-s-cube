import numpy as np
class agent():
    def __init__(self):
        self.Q_values = np.zeros([9,6,11]) #state size 9

        #https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb