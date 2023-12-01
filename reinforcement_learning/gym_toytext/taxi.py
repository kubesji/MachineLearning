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
env = gym.make("Taxi-v3", render_mode="ansi")
agent = Agent(env, epsilon_decay=0.99995, learning_rate=0.025)

# Initialise stuff needed for exploration and exploitation
# High number of episodes is needed to fully explore all possible combinations
EPISODES, TESTS = 50000, 50
history = deque(maxlen=100)

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]

    terminated, truncated = False, False
    steps = 0
    while not terminated and not truncated:
        action = agent.take_training_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)

        agent.fit(state, action, int(reward), terminated, truncated, new_state)

        state = new_state
        steps += 1

        if terminated and int(reward) > 0:
            history.append(steps)
            print(f"Episode {e:5d}: found the goal after {steps:3d}. ", end="")
            print(f"Average number of steps to reach goal is {sum(history)/len(history):6.2f}. ", end="")
            print(f"Current epsilon = {agent.eps:.3f}")
            agent.decay_epsilon()
            steps = 0


print("-------------- TEST --------------")
history.clear()
found = 0
for t in range(TESTS):
    state = env.reset()[0]
    steps, terminated, truncated = 0, False, False
    while not terminated and not truncated:
        action = agent.predict_action(state)

        new_state, reward, terminated, truncated, _ = env.step(action)

        state = new_state
        steps += 1

    if int(reward) > 0:
        found += 1
        history.append(steps)
        print(f"Test {t+1:4d}: found the goal after {steps:3d}. ", end="")
        print(f"Average number of steps to reach goal is {sum(history)/len(history):6.2f}. ")
    else:
        print(f"Test {t+1:4d}: did not find goal.")

print(f"-----------------\nPassenger reached destination {found} times out of {TESTS} rides.\n-----------------")
