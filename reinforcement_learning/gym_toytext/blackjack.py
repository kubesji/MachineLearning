import gymnasium as gym
import numpy as np
import random
from collections import deque, defaultdict


WIN = '\033[1m\033[92mwon\033[0m'
LOSS = '\033[1m\033[91mlost\033[0m'

class Agent:
    def __init__(self, environment, learning_rate=0.01, gamma=0.9):
        # Create action and observation spaces
        self.n_actions = environment.action_space.n
        self.observation_space = environment.observation_space
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.returns = defaultdict(lambda: [[] for _ in range(self.n_actions)])

        # Hyperparameters for actual learning
        self.gamma = gamma
        self.lr = learning_rate

    def predict_action(self, state):
        return np.argmax(self.Q[state])

    def take_training_action(self):
        # Take random action
        return random.randint(0, self.n_actions-1)

    def fit(self, episode):
        G = 0
        for s, a, r in episode:
            G = self.gamma * G + r
            self.returns[s][a].append(G)
            self.Q[s][a] += sum(self.returns[s][a]) / len(self.returns[s][a])


# Create environment
env = gym.make("Blackjack-v1")
agent = Agent(env, learning_rate=0.025, gamma=1)

# Initialise stuff needed for exploration and exploitation
# High number of episodes is needed to fully explore all possible combinations
EPISODES, TESTS = 2500000, 5000
history = deque(maxlen=1000)

print("------------ LEARNING ------------")
for e in range(EPISODES):
    state = env.reset()[0]

    terminated = False
    reward = 0
    episode = []
    while not terminated:
        action = agent.take_training_action()

        new_state, reward, terminated, truncated, _ = env.step(action)
        if truncated:
            print("WTF")
            exit(-1)

        episode.append((state, action, reward))

        state = new_state

    agent.fit(episode)

    history.append(1 if int(reward) > 0 else 0)

    if (e+1) % history.maxlen == 0:
        print(f"Episode {e+1:6d}/{EPISODES}: {sum(history)/len(history)*100:6.2f} % wins in last {len(history)} games.")
        steps = 0


print("-------------- TEST --------------")
history.clear()
found = 0
stats = {'wins': 0, 'losses': 0, 'draws': 0}
for t in range(TESTS):
    state = env.reset()[0]
    terminated = False
    while not terminated:
        action = agent.predict_action(state)

        new_state, reward, terminated, _, _ = env.step(action)

        state = new_state

    print(f"Test {t}: ", end="")
    if int(reward) > 0:
        print(f"Agent {WIN} ", end="")
        stats['wins'] += 1
    elif int(reward) < 0:
        print(f"Agent {LOSS} ", end="")
        stats['losses'] += 1
    else:
        print("Draw ", end="")
        stats['draws'] += 1
    print(f"with sum of {state[0]:2d} and {state[2]} usable aces, dealer holding {state[1]:2d}.")

print(f"----------------\nWins: {stats['wins']}, losses: {stats['losses']}, draws: {stats['draws']}\n----------------")

