import gym.spaces
import gym_Rubiks_Cube
import numpy as np

from model import DeepQNetwork


class rubik():
    def __init__(self):
        self.env = gym.make("RubiksCube-v0")
        self.state = self.env.reset()
        self.env.render()
        self.goal_face = 0
        self.goal_color = 0
        self.max = 0
        #self.shuffle()


    def get_face(self, i):  # 0 = middle
        return self.state[i * 9:(i + 1) * 9]

    def get_reward(self):
        goal_face = self.get_face(self.goal_face)
        i = 0
        for x in goal_face:
            if int(x)==  self.goal_color:
                i += 1
        self.max = np.max([i, self.max])
        return i - self.max


    def done(self):
        goal_face = self.get_face(self.goal_face)
        i = 0
        for x in goal_face:
            if int(x)==  self.goal_color:
                i += 1
        if i==len(goal_face):
            return True
        else:
            return False


    def shuffle(self):
        for i in range(0, 100):
            move = np.random.randint(0, 10)
            self.env.step(move)
            self.state = self.env.reset()

    def step(self,move):
        self.env.step(move)
        self.state = self.env.reset()
        reward = self.get_reward()
        done = self.done()
        observation = self.get_face(self.goal_face)
        self.env.render()
        return reward,self.state,done

import numpy as np
game = rubik()
print(game.get_face(0))
r,o,d= game.step(2)
print(o)
B = np.reshape(o, (6, 9,1))
print(B.shape)
network = DeepQNetwork(alpha=0.1)
print(network(o))