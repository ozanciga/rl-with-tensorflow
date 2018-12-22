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
MAX_NO_TRAIN_STEPS = 30
TARGET_Q_UPDATE_FREQUENCY = 10000


class Q:
    def __init__(self, env):
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]

        self.lr = 0.00025
        self.decay = 0.99
        self.momentum = 0.0
        self.eps = 1e-6

    def build_model(self):
        self.st = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.act = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.y = tf.placeholder(tf.float32, shape=[None])

        conv1 = tf.layers.conv2d(st, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)

        fc1 = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc1, 512)
        fc2 = tf.layers(fc1, self.n_actions)

        self.loss = tf.losses.mean_squared_error(labels, predictions)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.decay, momentum=self.momentum,
                                                   epsilon=self.eps)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, state):
        return sess.run(self.predictions, {self.x: state})

    def train(self, sess, state, action, y):
        _, loss = sess.run([self.train_op, self.loss], {self.st: state, self.act: action, self.y: y})
        return loss


'''

    # Create shared deep q network
    s, q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                      resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, name_scope="q-network")
    network_params = q_network.trainable_weights
    q_values = q_network(s)

    # Create shared target network
    st, target_q_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                      resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, name_scope="target-network")
    target_network_params = target_q_network.trainable_weights
    target_q_values = target_q_network(st)

    # Op for periodically updating target network with online network weights
    reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
'''


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

    def save(self, s, a, r, s1, terminal):
        self.memory.append((s.reshape(1, -1), a, r, s1.reshape(1, -1), terminal))

    def act(self, s):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space)
        else:
            q_values = self.model.predict(s)
            return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return  # not enough moves to do experience replay

        # online/primary network (q) = select an action given state (action_max)
        # target network (qt) = generate a Q-value for that action
        # action_max = argmax(q(next_state, a)) = via q network
        # qt <- reward+gamma*q(next_state, action_max)

        minibatch = random.sample(self.memory, BATCH_SIZE)
        x_batch, y_batch = np.zeros((BATCH_SIZE, self.observation_space), dtype=np.float32), np.zeros(
            (BATCH_SIZE, self.action_space), dtype=np.float32)
        for mj, (s, a, r, s1, terminal) in enumerate(minibatch):

            target = r if terminal else r + GAMMA*

            q_values[0][a] = q_star  # update action a's q-value

            x_batch[mj, ...] = s[0]
            y_batch[mj, ...] = q_values[0]

        self.model.fit(x_batch, y_batch, batch_size=BATCH_SIZE, verbose=False)

        self.exploration_rate = np.maximum(EXPLORE_MIN, self.exploration_rate * EXPLORATION_DECAY)


def preprocess_image(image, scope=None):
    with tf.name_scope(values=[image], name=scope, default_name='preprocess_image'):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.crop_to_bounding_box(image, 34, 0, 160, 160)
        image = tf.image.resize_nearest_neighbor(tf.expand_dims(image, axis=0), [84, 84])
        image = tf.to_float(image)
        image = tf.div(image, 255)
        image = tf.squeeze(image)
        return image


if __name__ == '__main__':
    # env.unwrapped.get_action_meanings() # prints ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    env = gym.make('BreakoutDeterministic-v4')
    #agent = DQNAgent(env, LEARNING_RATE)
    scores = deque(maxlen=100)

    global_step = 0  # counter to update target q-network parameters
for nj in range(1, NUM_EPISODES):
    state = env.reset()
    score, lives_left = 0, 5

    for _ in range(np.random.randint(1, MAX_NO_TRAIN_STEPS)):
        state, _, _, _ = env.step(1)

        state = tf.stack([preprocess_image(state)] * 4, axis=-1)
        state = tf.reshape([state], (1, 84, 84, 4))

    while True:
        action = agent.act(state)
        action = action + 1 if action <= 1 else 3
        s1, reward, terminal, _ = env.step(action)
        s1 = np.append(state[..., 1:], np.expand_dims(s1, 2), axis=2)

        # online/primary network (q) = select an action given state (action_max)
        # target network (qt) = generate a Q-value for that action
        # action_max = argmax(q(next_state, a)) = via q network
        # qt <- reward+gamma*q(next_state, action_max)

        q = online_network(state)
        action_max = np.argmax(q)
        qt = reward if terminal else reward + GAMMA*target_network(next_state, action_max)

        global_step += 1
        if global_step % TARGET_Q_UPDATE_FREQUENCY == 0:
            agent.copy_params()

        agent.save(state, action, reward, s1, terminal)  # for experience replay
        state = s1
        score += 1
        if terminal:
            scores.append(score)
            if nj % scores.maxlen == 0: print(
                'Episode: {}/{}, Avg. score (in last {} epis.): {}'.format(nj, NUM_EPISODES, scores.maxlen, np.mean(scores)))
            break  # died, episode ends

    agent.replay()  # experience replay (only place where we train!)
