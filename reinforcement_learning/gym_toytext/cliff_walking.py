import gymnasium as gym
import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, environment, learning_rate=0.01, gamma=0.9, epsilon_decay=0.99, epsilon_min=0.01):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space.n
        self.Q = np.random.rand(self.observation_space, self.n_actions)

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = 1
        self.eps_min = epsilon_min
        self.epc_decay = epsilon_decay

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.epc_decay)

    def predict_action(self, state):
        return np.argmax(self.Q[state, :])

    def take_training_action(self, state):
        # Take random action
        if random.random() <= self.eps:
            return random.randint(0, self.n_actions-1)
        # Predict best action
        else:
            return self.predict_action(state)

    def fit(self, state, action, reward, terminated, truncated, new_state):
        max_future_q = np.max(self.Q[new_state, :]) if not terminated else 0
        q_current = self.Q[state][action]

        new_q = (1 - self.lr) * q_current + self.gamma * self.lr * (reward + max_future_q)
        self.Q[state, action] = new_q



# Create environment
env = gym.make("CliffWalking-v0")
agent = Agent(env, epsilon_decay=0.95, learning_rate=0.025)

# Initialise stuff needed for exploration and exploitation
TESTS, MAX_STEPS = 50, 25000
history = deque(maxlen=10)

print("------------ LEARNING ------------")
state = env.reset()[0]
drops, steps = 0, 0
for s in range(MAX_STEPS):
    action = agent.take_training_action(state)

    new_state, reward, terminated, truncated, _ = env.step(action)
    reward = 100 if terminated else reward
    agent.fit(state, action, reward, terminated, truncated, new_state)

    state = new_state
    drops += 1 if reward == -100 else 0
    steps += 1

    if truncated:
        print("WTF")
        exit(-1)

    if terminated:
        history.append(steps)
        state = env.reset()[0]
        print(f"Found the goal after {steps}. Agent experienced {drops:3d} drops to the void since. ", end="")
        print(f"Average number of steps to reach goal is {sum(history)/len(history):6.2f}.", end="")
        print(f"Current epsilon = {agent.eps:.3f}")
        agent.decay_epsilon()
        steps, drops = 0, 0

print("-------------- TEST --------------")
history.clear()
for t in range(TESTS):
    state = env.reset()[0]
    steps, drops, terminated = 0, 0, False
    while not terminated:
        action = agent.predict_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state
        drops += 1 if reward == -100 else 0
        steps += 1

    history.append(steps)
    print(f"Found the goal after {steps}. Agent experienced {drops:3d} drops to the void since. ", end="")
    print(f"Average number of steps to reach goal is {sum(history) / len(history):6.2f}.")

