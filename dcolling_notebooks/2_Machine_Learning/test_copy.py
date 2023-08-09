import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

states = ["rock", "paper", "scissors"]
actions = ["rock", "paper", "scissors"]
num_episodes = int(1000)


# The QTable class is a data structure used for storing and accessing Q-values in reinforcement learning.
class QTable:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.q_table = {
            (state, action): 0 for state in self.states for action in self.actions
        }

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def update_qvalue(self, state, action, reward, alpha, gamma):
        old_q_value = self.get_q_value(state, action)
        next_max = max(
            [self.get_q_value(state, next_action) for next_action in self.actions]
        )
        next_max = 0
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
        self.q_table[(state, action)] = new_q_value


q_table = QTable(states, actions)


def user_choice():
    choice = random.choice(states)
    return choice


def comp_choice(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
    else:
        q_values = [q_table.get_q_value(state, action) for action in actions]
        action = actions[np.argmax(q_values)]
    return action


document_result = []


def get_state(user_choice, comp_choice):
    return (user_choice, comp_choice)


results_dict = {
    ("rock", "rock"): 0.0,
    ("rock", "paper"): -1.0,
    ("rock", "scissors"): 1.0,
    ("paper", "rock"): 1.0,
    ("paper", "paper"): 0.0,
    ("paper", "scissors"): -1.0,
    ("scissors", "rock"): -1.0,
    ("scissors", "paper"): 1.0,
    ("scissors", "scissors"): 0.0,
}


def reward(user_choice, comp_choice):
    result = results_dict[(user_choice, comp_choice)]
    document_result.append(result)
    return result


list_of_user_input = []
list_of_comp_input = []
contin_reward = []


def game_loop(alpha, gamma, epsilon, num_episodes):
    total_reward = 0
    for episode in range(num_episodes):
        # Decay the hyperparameters
        gamma *= g_decay_rate
        epsilon *= np.exp(-e_decay_rate * episode)

        state = user_choice()
        action = comp_choice(state, epsilon)

        # visualising results
        list_of_user_input.append(state)
        list_of_comp_input.append(action)
        reward_val = reward(state, action)
        total_reward += reward_val
        contin_reward.append(total_reward)

        q_table.update_qvalue(state, action, reward_val, alpha, gamma)
    return contin_reward


alpha = 0.3  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.5  # exploration rate
a_decay_rate = 1
g_decay_rate = 1
e_decay_rate = 1 / num_episodes


game_loop(alpha, gamma, epsilon, num_episodes)

x_var = np.linspace(0, num_episodes, num_episodes)
plt.plot(x_var, contin_reward)
plt.title("Reward Score Build-UP")
plt.xlabel("Number of Iterations")
plt.ylabel("Reward")
plt.show()

record = pd.DataFrame(
    {
        "Player's Choices": list_of_user_input,
        "Computer's Choices": list_of_comp_input,
        "Result": document_result,
    }
)
slice_result = record.iloc[-10:, :]
print(slice_result)
