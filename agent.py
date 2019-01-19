import numpy as np
import torch as T
from model import DeepQNetwork


class agent():
    def __init__(self,gamma,epsilon, alpha , maxMemSize, epsEnd, replace= 1000, actionSpace = [0,1,2,3,4,5,6,7,8,9,10]):
        self.Q_values = np.zeros([9,6,11]) #state size 9
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemSize
        self.steps = 0
        self.learn_step_counter= 0 #target network replacement
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt= replace
        self.Q_eval = DeepQNetwork(alpha=alpha)
        self.Q_next = DeepQNetwork(alpha=alpha)
        #https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb


    def storeTransition(self,state,action,reward, nextState):
        if self.memCntr < self.memSize:
            self.memory.append([state,action,reward,nextState])
        else:
            self.memory[self.memCntr%self.memSize] = [state,action,reward,nextState]
        self.memCntr += 1


    def chooseAction(self,observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1- self.EPSILON:
            action = T.argmax(actions)
        else:
            action = np.random.choice(actions)
        self.steps +=1
        return action

    def learn(self,batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and\
            self.learn_step_counter % self.replace_target_cnt ==0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.memCntr + batch_size < self.memSize:
            memStart = int(np.random.choice((range(self.memCntr))))
        else:
            memStart = int(np.random.choice(range(self.memCntr-batch_size)))
        minibatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(minibatch)

        Qpred = self.Q_eval.forward(list(memory[:,0]))
        Qnext = self.Q_next.forward(list(memory[:,3][:]))

        maxA = T.argmax(Qnext,dim=1).cuda()
        rewards = T.Tensor(list(memory[:,2])).cuda()
        Qtarget = Qpred
        Qtarget[:,maxA]= rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            self.EPSILON = np.max( [self.EPS_END, self.EPSILON -  1e-4] )
        loss = self.Q_eval.loss(Qtarget,Qpred).cuda()
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

ag = agent()