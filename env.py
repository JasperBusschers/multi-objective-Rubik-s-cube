import gym.spaces
import gym_Rubiks_Cube
import numpy as np

from model import DeepQNetwork


class rubik():
    def __init__(self, max_steps):
        self.env = gym.make("RubiksCube-v0")
        self.max_steps = max_steps
        self.state = self.env.reset()
        self.goal_face = 0
        self.goal_color = 0
        self.max = 0
        self.steps = 0
        self.shuffle()


    def get_face(self, i):  # 0 = middle
        return self.state[i * 9:(i + 1) * 9]

    def get_reward(self):
        goal_face = self.get_face(self.goal_face)
        i = 0
        for x in goal_face:
            if int(x)==  self.goal_color:
                i += 1
        return i


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
    def render(self):
        self.env.render()

    def shuffle(self):
        for i in range(0, 100):
            move = np.random.randint(0, 10)
            self.env.step(move)
            self.state = self.env.reset()

    def step(self,move):
        self.env.step(move)
        self.state = self.env.reset()
        reward = -1
        completed = self.done()
        done = completed
        self.steps += 1
        if self.steps >= self.max_steps or done:
            reward = self.get_reward() * 1500
            done = True
        self.state = self.env.reset()
        return self.state,reward, done , completed
