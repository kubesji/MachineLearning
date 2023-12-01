import gymnasium as gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from collections import deque
from keras.optimizers import Adam
from keras.utils import to_categorical


SUCCESS = '\033[1m\033[92msuccess\033[0m'
FAILURE = '\033[1m\033[91mfailure\033[0m'

class Agent:
    def __init__(self, env, learning_rate=0.01, gamma=0.9, epsilon_decay=0.99, epsilon_min=0.01, memory_length=2500,
                 batch_size=64):
        # Create action and observation spaces
        self.n_actions = env.action_space.n
        self.observation_space = env.observation_space.n

        # History and model itself
        self.memory = deque(maxlen=memory_length)
        self.model = self._create_model(learning_rate)

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.eps = 1
        self.eps_min = epsilon_min
        self.epc_decay = epsilon_decay
        self.batch_size = batch_size

    def _create_model(self, lr):
        model = Sequential()
        model.add(Input(shape=(self.observation_space, )))
        model.add(Dense(10, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(self.n_actions, activation='linear'))

        opt = Adam(learning_rate=lr)
        model.compile(loss="mse", optimizer=opt)
        return model

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.epc_decay)

    def predict(self, state):
        return self.model.predict(to_categorical(state, num_classes=self.observation_space).reshape(1, -1), verbose=0)

    def predict_action(self, state):
        return np.argmax(self.predict(state))

    def take_action(self, state):
        # Take random action
        if random.random() <= self.eps:
            return random.randint(0, self.n_actions-1)
        # Predict best action
        else:
            return self.predict_action(state)

    def save_experience(self, state, action, reward, terminated, truncated, new_state):
        self.memory.append((state, action, reward, terminated, truncated, new_state))

    def _sample_experience(self):
        return random.sample(self.memory, self.batch_size)

    def fit(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self._sample_experience()
        x, y = [], []
        for state, action, reward, terminated, truncated, new_state in batch:
            target_new = reward + (0 if terminated else self.gamma * np.amax(self.predict(new_state)))
            target = self.predict(state)[0]
            target[action] = target_new
            y.append(target)
            x.append(to_categorical(state, num_classes=self.observation_space))

        self.model.fit(np.array(x), np.array(y), epochs=1, verbose=0)


# Create environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
agent = Agent(env, epsilon_decay=0.9975)

# Initialise stuff needed for exploration and exploitation
EPISODES, MAX_STEPS = 2500, 200
history = deque(maxlen=100)

for e in range(EPISODES):
    state = env.reset()[0]
    reward = 0
    for s in range(MAX_STEPS):
        action = agent.take_action(state)

        new_state, reward, terminated, truncated, info = env.step(action)
        agent.save_experience(state, action, reward, terminated, truncated, new_state)
        state = new_state

        if truncated or terminated:
            print(f"Episode {e+1:4d} finished on step {s+1:2d} with {SUCCESS if reward == 1 else FAILURE}. ", end="")
            break

    history.append(int(reward))
    print(f"Success rate is {100*sum(history)/len(history):6.2f} %. Current epsilon = {agent.eps:.3f}")
    agent.fit()
    agent.decay_epsilon()
