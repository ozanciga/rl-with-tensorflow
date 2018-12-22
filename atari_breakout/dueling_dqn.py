'''
Dueling DQN
for Atari's breakout game.
Prioritized experience replay
-> some experience can teach
us more than the others.
'''

import gym
import tensorflow as tf
import numpy as np
from PIL import Image
from SumTree_ import Memory
from collections import deque

# PARAMS
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPISODES = 2000
GAMMA = 0.95
TARGET_Q_SYNC_FREQUENCY = 10000
EXPLORE_MAX = 1
EXPLORE_MIN = 0.01
EXPLORATION_DECAY = 0.995
MEMORY_CAPACITY = 200000
MAX_NO_TRAIN_STEPS = 30
FRAME_DIMS = (84, 84, 4)


def preprocess_frame(image):
    image = Image.fromarray(image)
    image = image.convert('L').crop((34, 0, 160, 160)).resize((84, 84), resample=Image.NEAREST)
    return np.array(image, dtype=np.float32) / 255.0


# Q(s,a)=V(s)+(A(s,a)-avg. advantage) -> decouple value of being in
# state and the advantage A of taking action a
class Q:
    def __init__(self, n_actions, frame_dims, name):
        self.scope = name
        self.action_space = n_actions

        with tf.variable_scope(name):
            self.exploration_rate = EXPLORE_MAX
            self.state = tf.placeholder(tf.float32, [None, *frame_dims])
            self.action = tf.placeholder(tf.float32, [None, self.action_space])
            self.q_target = tf.placeholder(tf.float32, [None])  # for error calc./priority estimation.
            self.ISWeights = tf.placeholder(tf.float32, [None, 1])

            self.conv1 = tf.layers.conv2d(self.state, 32, 8, 4, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.conv2 = tf.layers.conv2d(self.conv1, 64, 4, 2, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.conv3 = tf.layers.conv2d(self.conv2, 64, 3, 1, activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.flatten = tf.layers.flatten(self.conv3)

            self.value_fc1 = tf.layers.dense(self.flatten, 512, activation=tf.nn.relu,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value = tf.layers.dense(self.value_fc1, 1,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())  # activation=None/linear

            self.advantage_fc1 = tf.layers.dense(self.flatten, 512, activation=tf.nn.relu,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.advantage = tf.layers.dense(self.advantage_fc1, self.action_space,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.q_temp = self.value + (self.advantage - tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            #self.q_pred = tf.boolean_mask(self.q_pred, self.action)            #self.q_pred = tf.gather(self.q_pred, tf.where(tf.equal(tf.constant(1, dtype=tf.float32))))
            self.q_pred = tf.reduce_sum(tf.multiply(self.q_temp, self.action), axis=1)

            #self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_pred))
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q_pred)
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_pred))

            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)

    def select_action(self, sess):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space)
        else:
            q = sess.run(self.q_temp, feed_dict={self.state: state})
            return np.argmax(q)


class DuelingDQN:
    def __init__(self, n_actions, frame_dims):
        self.online_q_network = Q(n_actions, frame_dims, name='online_q_network')
        self.target_q_network = Q(n_actions, frame_dims, name='target_q_network')
        self.memory = Memory(MEMORY_CAPACITY)
        self.action_space = n_actions
        self.exploration_rate = EXPLORE_MAX

    # function credit: github@dennybritz/reinforcement-learning
    def sync_networks(self, sess):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(self.online_q_network.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(self.target_q_network.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        sess.run(update_ops)

    def replay(self):
        if len(self.memory.tree) < BATCH_SIZE:  #
            return
        '''
        online network selects action_max (=argmax(Q(s,a))
        target network outputs Q(next_state, action_max)
        '''

        states, actions, targets, errors = [], [], [], []
        indices, minibatch, ISWeights = self.memory.sample(BATCH_SIZE)
        for mj in range(BATCH_SIZE):
            s, a, r, s1, terminal = minibatch[mj]
            idx = indices[mj]
            action_max = self.online_q_network.select_action(s)
            if terminal:
                target = reward
            else:
                q = sess.run(self.target_q_network.q_temp, feed_dict={self.target_q_network.state: np.reshape(s1, (1, 84, 84, 4))})
                target = reward + GAMMA * q[0][action_max]

            q = sess.run(self.online_q_network.q_temp, feed_dict={self.online_q_network.state: np.reshape(s, (1, 84, 84, 4))})
            error = np.abs(target - q[0][a])  # abs(q(s,a)-target)
            error = np.minimum(error, 1) # clip error to 1
            self.memory.tree.update(idx, error)

            states.append(s)
            actions.append(a)
            targets.append(target)
            errors.append(error)

        _, loss, absolute_errors = sess.run(
            [self.online_q_network.optimizer, self.online_q_network.loss, self.online_q_network.abs_errors],
            feed_dict={
                self.online_q_network.state: np.array(states),
                self.online_q_network.q_target: np.array(targets),
                self.online_q_network.action: np.identity(self.action_space)[np.array(actions)],
                self.online_q_network.ISWeights: ISWeights
            })

        self.exploration_rate = np.maximum(EXPLORE_MIN, self.exploration_rate * EXPLORATION_DECAY)


if __name__ == '__main__':
    env = gym.make('BreakoutDeterministic-v4')  # env outputs every 4th frame, for rapid learning
    n_actions, frame_dims = env.action_space.n, FRAME_DIMS
    agent = DuelingDQN(n_actions, frame_dims)
    scores = deque(maxlen=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        agent.sync_networks(sess)

        global_step = 0
        for epis in range(NUM_EPISODES):
            state = env.reset()
            for _ in range(np.random.randint(1, MAX_NO_TRAIN_STEPS)):
                state, _, _, _ = env.step(1)

            state = np.stack([preprocess_frame(state)] * 4, axis=-1)

            total_steps_alive = 0
            while True:
                global_step += 1
                if global_step % TARGET_Q_SYNC_FREQUENCY:
                    agent.sync_networks(sess)

                action_max = agent.online_q_network.select_action(np.expand_dims(state, 0))
                next_state, reward, terminal, info = env.step(action_max)  # s1= next state
                next_state = np.append(state[..., 1:], np.reshape(preprocess_frame(next_state), (84, 84, 1)), axis=-1)

                agent.memory.store((state, action_max, reward, next_state, terminal))
                state = next_state
                total_steps_alive += 1
                if terminal:
                    scores.append(total_steps_alive)
                    if epis % scores.maxlen == 0:
                        print('Episode: {}/{}, Avg. score (in last {} epis.): {}'.format(epis, NUM_EPISODES, scores.maxlen, np.mean(scores)))
                    break

                agent.replay()
