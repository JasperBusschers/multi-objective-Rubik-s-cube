from agent import agent
from env import rubik
from utils import plotLearning
import numpy as np

def train():
    env = rubik(max_steps= 9000)
    ag = agent( gamma = 0.95 , epsilon = 1.0,epsEnd=0.05,
                alpha = 0.003, maxMemSize=50000,
                replace=None)

    while ag.memCntr < ag.memSize:
        observation = env.env.reset()
        done = env.done()
        while not done:
            action = np.random.choice(ag.actionSpace)
            nextObservation, reward, done , completed = env.step(action)
            if done and not completed:
                reward -= 100
            if done and completed:
                reward += 1000
            ag.storeTransition(observation,action,reward,nextObservation)
            observation = nextObservation
        print('done initialising memory')

        scores = []
        epsHistory = []
        numGames = 32000
        batch_size = 32


        for i in range(numGames):

            epsHistory.append(ag.EPSILON)
            done = False
            observation = env.env.reset()
            if i%1000 == 0:
                print('starting game  ' + str(i + 1) + ' epsilon = ' + str(ag.EPSILON))
                print('pre setup')
                env.render()
            score = 0
            while not done:
                action = ag.chooseAction(observation)
                nextObservation, reward, done, completed = env.step(action)
                score += reward
                if done and not completed:
                    reward -= 100
                if done and completed:
                    reward += 1000
                ag.storeTransition(observation, action, reward, nextObservation)
                observation = nextObservation
                ag.learn(batch_size)
            if i%1000 == 0:
                print('post setup ')
                print('score = ' + str(score))
                env.render()
            scores.append(score)
        x = [i + 1 for i in range(numGames)]
        fileName = str(numGames) + 'Games' + 'Gamma' + str(ag.GAMMA) + \
                   'Alpha' + str(ag.ALPHA) + 'Memory' + str(ag.memSize) + '.png'
        plotLearning(x, scores, epsHistory, fileName)

train()