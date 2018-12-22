# i roughly follow these tutorials:
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb

import numpy as np
import gym
import time

env = gym.make('FrozenLake-v0')

n_act, n_st = env.action_space.n, env.observation_space.n

n_episodes = 15000
max_explore_prob, explore_prob = [0.5]*2  # instead of optimal action, take a random action w/ given prob
lr = 0.8  # alpha in q update/learning rate
gamma = 0.95 # discount rate

Q = np.zeros((n_st, n_act))
for epis in range(n_episodes):
    s = env.reset()  # puts you top left on 4x4 grid
    # walk around the grid for a finite # steps for each episode
    for _ in range(1000):
        # choose an action: optimal or explore/probabilistic action
        act = np.random.randint(0, n_act) if (np.random.rand() <= explore_prob) else np.argmax(Q[s, :])
        # move into the direction of act
        s_new, reward, ended, _ = env.step(act)  # state, reward, are we done(reached goal or dead)?
        # update
        Q[s, act] = Q[s, act]*(1-lr) + lr*(reward + gamma*np.max(Q[s_new, :]))
        s = s_new
        if ended: break
    # update the exploration prob. (more exploration at the beginning)
    explore_prob = np.exp(-(0.001)*epis)*max_explore_prob


# we must've learned by now, let's make the agent play the game
s = env.reset()
ended = False
while not ended:
    act = np.argmax(Q[s, :])
    s_new, _, ended, _ = env.step(act)
    env.render()
    time.sleep(0.5)  # for visualization
    s = s_new

