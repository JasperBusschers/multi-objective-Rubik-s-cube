import torch
import gym
import numpy as np
import gym.spaces
import gym_Rubiks_Cube
from GAIL_TEST.model import GAIL
from env import rubik
import matplotlib.pyplot as plt


def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v2"
    # env_name = "LunarLanderContinuous-v2"
    solved_reward = 300
    random_seed = 0
    max_timesteps = 1000  # max time steps in one episode
    n_eval_episodes = 20  # evaluate average reward over n episodes
    lr = 0.0002  # learing rate
    beta1 = 0.5  # beta 1 for adam optimizer
    n_epochs = 10000  # number of epochs
    n_iter = 300  # updates per epoch
    batch_size = 30  # num of transitions sampled from expert
    directory = "./{}".format(env_name)  # save trained models
    filename = "GAIL_{}_{}".format(env_name, random_seed)
    ###################################
    # lr = 0.0002 beta1 = 0.5 n_iter = 100 batch size = 100
    #
    #
    #
    #
    #
    ####

    env = rubik(100)
    state_dim = 54
    action_dim = 12
    max_action = 11
    print("eeeeee")
    policy = GAIL(env_name, state_dim, action_dim, max_action, lr, beta1)
    print("eup")
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # training procedure
    print('training has started')
    averages = []
    for epoch in range(1, n_epochs + 1):
        # update policy n_iter times
        loss_policy, loss_agent = policy.update(n_iter, batch_size)
        print('-------------------- epoch: '+str(epoch)+'-------------------------------------')
        print('average discriminator loss ' + str(loss_policy))
        print('average agent loss ' + str(loss_agent))
        print(epoch)
        # evaluate in environment
        total_reward = 0
        for episode in range(n_eval_episodes):
            state = env.shuffle()
            for t in range(max_timesteps):
                action = policy.select_action(state)
                action = np.argmax(action)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break

        avg_reward = int(total_reward / n_eval_episodes)
        averages.append(avg_reward)
        # print("############################")
        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))
        # print("############################")

        if avg_reward > 5:
            print("########### Solved! ###########")
            policy.save(directory, filename)
            break
    # plot the scores
    scores = avg_reward
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title('simple reward scheme , state size = 9')
    plt.ylabel('Mean Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    train()