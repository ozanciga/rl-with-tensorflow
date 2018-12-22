import tensorflow as tf
from collections import deque
import gym
import random
import numpy as np

EXPLORE_MAX = 1
EXPLORE_MIN = 0.01
EXPLORATION_DECAY = 0.995
MAX_MEMORY = 100000
BATCH_SIZE = 64
NUM_EPISODES = 20000
GAMMA = 0.95
LEARNING_RATE = 0.001

# Set Eager API
#tf.enable_eager_execution()    # uncomment for eager execution (slower)

class DQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(24, activation='relu')
        self.layer2 = tf.keras.layers.Dense(24, activation='relu')
        self.layer3 = tf.keras.layers.Dense(action_space, activation='linear')

    def call(self, x, training=None, mask=None):
        return self.layer3(self.layer2(self.layer1(x)))


class DQNAgent:
    def __init__(self, env, learning_rate):
        self.env = env
        self.action_space = env.action_space.n
        self.observation_space = int(env.observation_space.shape[0])

        self.exploration_rate = EXPLORE_MAX
        self.explore_min = EXPLORE_MIN

        # self.grad = tfe.implicit_gradients(self.loss_fn) # cleaning up old api
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.model = DQN(self.action_space)
        self.model.compile(optimizer=self.optimizer, loss='mse')

        self.memory = deque(maxlen=MAX_MEMORY)

    '''
    def loss_fn(self, model_fn, inputs, labels):
        return tf.reduce_mean(tf.losses.mean_squared_error(labels, model_fn(inputs)))

    def train(self, x, y):
        self.optimizer.apply_gradients(self.grad(self.model, x, y))
    '''

    def save(self, s, a, r, s1, terminal):
        self.memory.append((s.reshape(1, -1), a, r, s1.reshape(1, -1), terminal))

    def act(self, s):
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample() # np.random.randint(0, self.action_space)
        else:
            q_values = self.model.predict(s)
            return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return  # not enough moves to do experience replay

        minibatch = random.sample(self.memory, BATCH_SIZE)
        x_batch, y_batch = np.zeros((BATCH_SIZE, self.observation_space), dtype=np.float32), np.zeros((BATCH_SIZE, self.action_space), dtype=np.float32)
        for mj, (s, a, r, s1, terminal) in enumerate(minibatch):
            # update q value of this (s,a,r,s) with the latest parameters of the network
            # (replayed action record may be very early)
            q_star = r if terminal else r + (GAMMA * np.amax(self.model.predict(s1)[0]))
            # use network to get best action of this state s1
            q_values = self.model.predict(s)
            q_values[0][a] = q_star  # update action a's q-value

            x_batch[mj, ...] = s[0]
            y_batch[mj, ...] = q_values[0]

        self.model.fit(x_batch, y_batch, batch_size=BATCH_SIZE, verbose=False)

        self.exploration_rate = np.maximum(EXPLORE_MIN, self.exploration_rate*EXPLORATION_DECAY)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env, LEARNING_RATE)
    scores = deque(maxlen=100)

    for nj in range(1, NUM_EPISODES):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state.reshape(1, -1))  # take step according to optimal policy
            s1, reward, terminal, _ = env.step(action)  # s1= next state

            agent.save(state, action, reward, s1, terminal)  # for experience replay
            state = s1
            score += 1
            if terminal:
                scores.append(score)
                if nj % scores.maxlen == 0:
                    print('Episode: {}/{}, Avg. score (in last {} epis.): {}, Exploration rate: {:.2f}'.format(nj, NUM_EPISODES, scores.maxlen, np.mean(scores), agent.exploration_rate))
                break  # died, episode ends

        agent.replay()  # experience replay to do smart exploration (as opposed to random search)
